__precompile__()

using Distributed
@everywhere import PyCall 
@everywhere using StatsBase: sample,Weights,percentile
@everywhere using IterativeSolvers: lsmr
@everywhere using SparseArrays: sparse
@everywhere using NearestNeighbors#: KDTree,knn,BallTree
@everywhere using HDF5#: h5read,h5write,h5open
@everywhere using JuliaDB
@everywhere using LinearAlgebra
@everywhere using ProgressMeter
@everywhere using DelimitedFiles
@everywhere using Distributions: Gamma
@everywhere GC.gc()

@everywhere taup = PyCall.pyimport("obspy.taup")
@everywhere model = taup.TauPyModel(model="ak135")
@everywhere get_ray_paths_geo = model.get_ray_paths_geo

@everywhere const HVR = 1.0
@everywhere const EARTH_CMB = 3481.0
@everywhere const EARTH_RADIUS = 6371.0


@everywhere using LinearAlgebra: cross,dot
@everywhere function gcp(lat1::T,lon1::T,lat2::T,lon2::T,delta::T) where T <: Real
    lat1 = deg2rad(lat1)
    lon1 = deg2rad(lon1)
    lat2 = deg2rad(lat2)
    lon2 = deg2rad(lon2)

    ur1 = [cos(lat1)*sin(lon1),sin(lat1),cos(lat1)*cos(lon1)]
    ur2 = [cos(lat2)*sin(lon2),sin(lat2),cos(lat2)*cos(lon2)]

    norm_vec = cross(ur1,ur2)
    unorm = norm_vec./sqrt(sum(norm_vec.^2))
    
    tvec = cross(unorm,ur1)
    utvec = tvec./sqrt(sum(tvec.^2))
    total_arc = acos(dot(ur1,ur2))

    r0 = 6371.0f0
    angs2use = collect(range(0.0f0, stop = total_arc, length = convert(Int64,ceil(total_arc*r0/delta))))
    m = length(angs2use)

    unit_r = (ones(m,1)*reshape(ur1,1,3)).*transpose(ones(3)*reshape(cos.(angs2use),1,m)) +
             (ones(m,1)*reshape(tvec,1,3)).*transpose(ones(3)*reshape(sin.(angs2use),1,m))

    unit2 = @view unit_r[:,2]
    id = findall(x->x>1.0f0,unit2)
    unit2[id].=1.0f0
    id = findall(x->x<-1.0f0,unit2)
    unit2[id].=-1.0f0
    lats = asin.(unit_r[:,2])
    lons = atan.(unit_r[:,1],unit_r[:,3])

    path = hcat(lats,lons)
    dist = total_arc*r0
    #dist = [total_arc*r0,r0]
    #dist = reshape(dist,1,2)
    #return vcat(path,dist)
    return path,dist
end

@everywhere function wrap_get_ray_paths_geo(evdep::Float32,evlat::Float32,
                        evlon::Float32,stlat::Float32,stlon::Float32,
                        phase_list::Array{String,1},sampleds::Float32)
    arr = get_ray_paths_geo(evdep,evlat,evlon,stlat,stlon,phase_list,sampleds)
    rayparam = 0.0
    raytakeoffangle = 0.0
    ifray = true
    raypts = Float32[]
    if length(arr)>0 
        lon = get(arr[1].path,"lon")   
        lon = deg2rad.(lon)
        lat = get(arr[1].path,"lat")   
        lat = pi/2.0 .- deg2rad.(lat)
        dep = get(arr[1].path,"depth")   
        rad = EARTH_RADIUS .- dep .* HVR
        raypts = hcat(rad,lat,lon)
        rayparam = arr[1].ray_param
        raytakeoffangle = deg2rad(arr[1].takeoff_angle)
    else 
        ifray = false
    end
    GC.gc()
    return (ifray,rayparam,raytakeoffangle,raypts)
end

@everywhere function subspaceinv(datasub::IndexedTable,iter::Int64,
                    ncells::Int64,phases::Array{Array{String,1},1},joint::Int64,
                    periods::Array{Float64,1},dispersyn::Array{Float64,1},
                    lnvp::Array{Float64,2},lnvs::Array{Float64,2},surfdata::IndexedTable)
    mindist = 2.0f0
    threshold = 0.2
    weight_s = 0.1
    nlat = 512
    nlon = 1024
    nrad = 128
    sparsefrac = 0.01f0#0.003f0
    k = 1
    columnnorm = 0
   
    cellsph = generate_vcells(ncells)
    sph2xyz!(cellsph)
    kdtree = KDTree(transpose(cellsph);leafsize=10)
    #kdtree = BallTree(transpose(cellxyz), Euclidean(),leafsize = 10)
    
    dataidx = Int32(0)
    ndatabody = length(datasub)
    ndatasurf = length(surfdata)
    ndata = ndatabody+ndatasurf
    @info "start sensitivity mastrix $(ndata)"

    maxnonzero = Int32(sparsefrac*ncells*ndata)
    row = zeros(Int32,maxnonzero)
    col = zeros(Int32,maxnonzero)
    nonzerosall = zeros(Float32,maxnonzero)
    b = zeros(Float32,ndata)

    evlatall = select(datasub,:evlat)
    evlonall = select(datasub,:evlon)
    evdepall = select(datasub,:evdep)
    stlatall = select(datasub,:stlat)
    stlonall = select(datasub,:stlon)
    azimall =  select(datasub,:azim)
    eventidnewall = select(datasub,:eventidnew)
    phaseall = select(datasub,:phase)
    issall = select(datasub,:iss)
    datasuball = select(datasub,:dres)

    zeroid = 0
    colidloc = zeros(Int64,4)
    rowidloc = zeros(Int64,4)
    nonzerosloc = zeros(Float64,4)
    rowray = zeros(Float64,ncells)

    #@showprogress for ii in 1:ndata
    for ii in 1:ndatabody
        evlat = evlatall[ii]
        evlon = evlonall[ii]
        evdep = evdepall[ii]
        stlat = stlatall[ii]
        stlon = stlonall[ii]
        azimulth = azimall[ii]
        srcidx = eventidnewall[ii]
        phaseno = phaseall[ii]
        iss =  issall[ii]
        bres = datasuball[ii]
        phasew = phases[phaseno]

        ray = wrap_get_ray_paths_geo(evdep, evlat, evlon, stlat, 
                                    stlon, phasew, mindist)
        ifray = ray[1]
        if !ifray 
            continue
            @info "$(ii)th data"
        end
        rayparam = ray[2]
        raytakeoffangle = ray[3]
        raysph = ray[4]

        dataidx += oneunit(dataidx)
        sph2xyz!(raysph)
        nseg = size(raysph,1)
        raydiffsub = diff(raysph,dims=1)
        rayseg = sqrt.(raydiffsub[:,1].^2+
                                  raydiffsub[:,2].^2+
                                  raydiffsub[:,3].^2)

        rayidx, _ = knn(kdtree, transpose(raysph), 1, false)
        idxssub = [x[1] for x in rayidx]
        rowray .= 0.0f0
        @inbounds for ii = 1:nseg-1
        rowray[idxssub[ii]] += rayseg[ii]
        end
        colid = findall(x->x>0.0f0,rowray)
        
        weight = 1.0/(1+0.05*exp(bres^2*threshold))
        b[dataidx] = bres*weight
        nonzeros = rowray[colid].*weight
        nnzero = length(colid)
        colid = colid .+ iss*ncells
        
        colidloc .= 0
        rowidloc .= 0
        nonzerosloc .= 0.0
        deltar = 10.0
        colidloc[1] = 2*ncells+srcidx*4+1
        rowidloc[1] = dataidx
        nonzerosloc[1] = rayparam/tan(raytakeoffangle)/EARTH_RADIUS*deltar
        
        colidloc[2] = 2*ncells+srcidx*4+2
        rowidloc[2] = dataidx
        nonzerosloc[2] = -rayparam*cos(azimulth)/EARTH_RADIUS*deltar
        colidloc[3] = 2*ncells+srcidx*4+3
        rowidloc[3] = dataidx
        nonzerosloc[3] = rayparam*sin(azimulth)*cos(deg2rad(evlat))/
                         EARTH_RADIUS*deltar
        colidloc[4] = 2*ncells+srcidx*4+4
        rowidloc[4] = dataidx
        nonzerosloc[4] = 1.0
        
        colid = convert(Array{Int32,1},vcat(colid,colidloc))
        rowid = ones(Int32,nnzero+4) .* dataidx
        nonzeros = convert(Array{Float32,1},vcat(nonzeros,nonzerosloc)*weight)

        col[zeroid+1:zeroid+nnzero+4] = colid
        row[zeroid+1:zeroid+nnzero+4] = rowid
        nonzerosall[zeroid+1:zeroid+nnzero+4] = nonzeros
        zeroid = zeroid+nnzero+4
    end

    if joint == 1
        weightsurf = 10.0
        delta = 5.0f0
        cutdep = 17
        #ndatasurf = length(surfdata)
        @info "begin surface wave $(ndatasurf)"
        evlatsurf = select(surfdata,:evlat)
        evlonsurf = select(surfdata,:evlon)
        stlatsurf = select(surfdata,:stlat)
        stlonsurf = select(surfdata,:stlon)
        periodssurf = select(surfdata,:period)
        dispersion = select(surfdata,:disper)
        #periods = select(periods,:periods)
        #dispersyn = select(dispersyn,:disper)

        drad = (EARTH_RADIUS-EARTH_CMB)/nrad
        pathrad = EARTH_RADIUS .- HVR.*(drad/2.0:drad:nrad*drad-drad/2.0)
        pathrad = collect(Float32,pathrad)
        #@showprogress for idata = 1:ndatasurf
        for idata = 1:ndatasurf
            dataidx += oneunit(dataidx)
            evlat = evlatsurf[idata]
            evlon = evlonsurf[idata]
            stlat = stlatsurf[idata]
            stlon = stlonsurf[idata]
            pathlatlon,dist = gcp(evlat,evlon,stlat,stlon,delta)
            #dist = pathlatlon[end,1]
            pathlat = @view pathlatlon[1:end,1]
            pathlon = @view pathlatlon[1:end,2]
            pathrad = pathrad[1:cutdep]

            npts = size(pathlat,1)
            idx = zeros(Int64,npts*cutdep)
            depidx = zeros(Int64,npts*cutdep)

            for idep = 1:cutdep
                pathradlay = pathrad[idep]
                xcor = pathradlay.*sin.(pi/2 .-pathlat).*cos.(pathlon)
                ycor = pathradlay.*sin.(pi/2 .-pathlat).*sin.(pathlon)
                zcor = pathradlay.*cos.(pi/2 .-pathlat)
                idxlay, _ = knn(kdtree, transpose(hcat(xcor,ycor,zcor)), 1, false)
                idxlay = [x[1] for x in idxlay]
                idx[(idep-1)*npts+1:idep*npts] = idxlay
                depidx[(idep-1)*npts+1:idep*npts] .= idep
            end

            uidx = unique(idx)
            nnzero = length(uidx)
            period = periodssurf[idata]
            peridx = findfirst(x->x==period,periods)
            disperobs = dispersion[idata]
            #ressurf = (dist/disperobs*1000 - dist/dispersyn[peridx])
            ressurf = disperobs/1000.0f0-dispersyn[peridx]
            weight = 1.0/(1+0.05*exp(ressurf^2*0.1))*weightsurf
            b[dataidx] = weight*ressurf
            #@info idata,evlat,evlon,stlat,stlon,weight*ressurf,nnzero,period,peridx
            for ii = 1:nnzero
                zeroid = zeroid+1
                col[zeroid] = uidx[ii]+ncells
                row[zeroid] = dataidx
                idxt = findall(x->x==uidx[ii],idx)
                gtmp = 0.0
                for jj in idxt
                    #gtmp += lnvs[depidx[jj],peridx]
                    gtmp += delta*lnvs[depidx[jj],peridx]
                end

                #nzero_value = -delta*gtmp/dispersyn[peridx]^2*weight
                nzero_value = gtmp/dist*weight
                nonzerosall[zeroid] = nzero_value

                zeroid = zeroid+1
                col[zeroid] = uidx[ii]
                row[zeroid] = dataidx
                gtmp = 0.0
                for jj in idxt
                    #gtmp += lnvp[depidx[jj],peridx]
                    gtmp += delta*lnvp[depidx[jj],peridx]
                end
                #nzero_value = -delta*gtmp/dispersyn[peridx]^2*weight
                nzero_value = gtmp/dist*weight
                nonzerosall[zeroid] = nzero_value
            end
        end
    end

    @info "Finishing ray tracing with nonzeros: $(zeroid)"
    b = b[1:dataidx]
    col = col[1:zeroid]
    row = row[1:zeroid]
    nonzerosall = nonzerosall[1:zeroid]
    G = sparse(row,col,nonzerosall)

    row = nothing
    col = nothing
    nonzerosall = nothing
    
    ncol = size(G,2)
    if columnnorm == 1
    cnorm = zeros(Float32,ncol)
    @inbounds for icol = 1 : ncol
        i = G.colptr[icol]
        k = G.colptr[icol+1] - 1
        n = i <= k ? norm(G.nzval[i:k]) : 0.0  
        n > 0.0 && (G.nzval[i:k] ./= n)
        cnorm[icol] = n
    end
    colid = findall(x->x>0,cnorm)# .+ iss*ncells
    normthresh = percentile(cnorm[colid],20)
    colid = findall(x->x>normthresh,cnorm)# .+ iss*ncells
    G = G[:,colid]
    cnorm = cnorm[colid]
    end

    damp = 10.0
    atol = 1e-4
    btol = 1e-6
    conlim = 100
    maxiter = 50
    x = lsmr(G, b, ¿ = damp, atol = atol, btol = btol,
	     maxiter = maxiter, log = true)
    #x = lsmr(G,b,Î»=damp, atol = atl, btol = btol, 
    #         maxiter = maxiter,log = true)
    @info x[2]
    @info "max col no.$(length(colid))"

    x = x[1]
    xall = zeros(Float32,ncol)
    if columnnorm == 1
    x ./= cnorm
    x = convert(Array{Float32,1},x)
    xall[colid] = x
    else
    x = convert(Array{Float32,1},x)
    xall = x
    end
    xp = xall[1:ncells]
    xs = xall[ncells+1:2*ncells]
    @info "min and max vp and vs",minimum(xp),maximum(xp),minimum(xs),maximum(xs)
    xall = nothing

    @info "begin projection matrix"
    dlat = pi/nlat
    dlon = 2*pi/nlon
    drad = (EARTH_RADIUS-EARTH_CMB)/nrad
    lat = -(pi-dlat)/2.0:dlat:(pi-dlat)/2.0
    lat = collect(Float32,pi/2.0 .- lat)
    lon = dlon/2.0:dlon:2*pi
    lon = collect(Float32,lon)
    rad = EARTH_RADIUS .- HVR.*(drad/2.0:drad:nrad*drad-drad/2.0)
    rad = collect(Float32,rad)
    npara = nlat*nlon*nrad

    gridxyz = zeros(Float32,npara,3)
    colgp = ones(Int32,npara)

    idx = 0
    for xj in lat
        for yj in lon
            for zj in rad
                idx += 1
                gridxyz[idx,1] = zj * sin(xj) * cos(yj)
                gridxyz[idx,2] = zj * sin(xj) * sin(yj)
                gridxyz[idx,3] = zj * cos(xj) 
            end
        end
    end
    @info "Finishing projection matrix"
    
    k = 1
    colgp, _ = knn(kdtree, transpose(gridxyz), k, false)
    gridxyz = nothing
    colgp = [Int32(x[1]) for x in colgp]
    
    rowgp = collect(Int32,1:npara)
    valuegp = ones(Float32,npara)
    Gp = sparse(rowgp,colgp,valuegp,npara,ncells)

    rowgp = nothing
    colgp = nothing
    valuegp = nothing

    vp = Gp*xp
    vs = Gp*xs
    h5open("juliadata/eofe_vp$(iter).h5","w") do file
        write(file,"vp",vp)
    end
    h5open("juliadata/eofe_vs$(iter).h5","w") do file
        write(file,"vs",vs)
    end
    return nothing
end


##
#"""
#generate vcells
#"""
#
@everywhere function generate_vcells(ncells::Number=20000)::Array{Float32,2}
    cellphi = acos.(rand(ncells).*2 .- 1.0)# .- pi/2.0
    celltheta = 2.0*pi*rand(ncells)
    stretchradialcmb = ( EARTH_RADIUS - EARTH_CMB ) .* HVR
    #cellrad =  stretchradialcmb .* rand(ncells) .+ 
    #            EARTH_RADIUS .- stretchradialcmb
    ncells_base = 5000
    cellrad_base =  stretchradialcmb .* rand(ncells_base) .+ 
                    EARTH_RADIUS .- stretchradialcmb
    distrib = Gamma(2.0,200.0)
    cellrad_refine = EARTH_RADIUS .- rand(distrib,ncells-ncells_base)
    cellrad = vcat(cellrad_base,cellrad_refine)
    #cellrad =  stretchradialcmb .* rand(ncells*10) .+ 
    #            EARTH_RADIUS .- stretchradialcmb
    #cellrad = sample(cellrad, Weights(cellrad.^2), ncells,replace=false)
    cellptssph = hcat(cellrad,cellphi,celltheta)
    cellptssph = convert(Array{Float32,2},cellptssph)
    return cellptssph
end
##

#"""
#transform geo-coordinates to spherical coordinates
#Args
#lat,lon,dep
#"""
@everywhere function geo2sph!(geopts::Array{T,2}) where T <: Real
    npoints = size(geopts,1)
    for ii = 1:npoints
    lat = geopts[ii,1]
    lon = geopts[ii,2]
    depth = geopts[ii,3]
    geopts[ii,1] = -depth .+ EARTH_RADIUS 
    geopts[ii,2] = pi/.2 .- deg2rad.(lat) 
    geopts[ii,3] = deg2rad.(lon)
    end
end

##

##
#"""
#transform spherical coordinates to Cartician
#Args
#rad,phi,theta
#"""
@everywhere function sph2xyz!(sph::Array{T,2}) where T <: Real
    npoints = size(sph,1)
    for ii = 1:npoints
    rad = sph[ii,1]
    phi = sph[ii,2]
    theta = sph[ii,3]
    sph[ii,1] = rad * sin(phi) * cos(theta)
    sph[ii,2] = rad * sin(phi) * sin(theta)
    sph[ii,3] = rad * cos(phi) 
    end
end
##

##
#"""
#sample events based on their distribution
#"""
##
@everywhere function geteventidx(eventsloc::Array{Float32,2},
                                eventcell::Number = 100000)::Array{Int32,1}
    cellphi = acos.(rand(eventcell).*2 .- 1.0)
    celltheta = 2.0*pi*rand(eventcell)
    cellrad = EARTH_RADIUS .- 660.0 * rand(eventcell) 

    sph = hcat(cellrad,cellphi,celltheta)
    sph = convert(Array{Float32,2},sph)

    #cellxyz = sph2xyz(sph)
    sph2xyz!(sph)
    kdtree = KDTree(transpose(sph);leafsize=10)
    #sph = geo2sph(eventsloc)
    geo2sph!(eventsloc)
    sph2xyz!(eventsloc)
    k = 1
    idxs, _ = knn(kdtree, transpose(eventsloc), k, false)
    idxs = [x[1] for x in idxs]
    idxs = convert(Array{Int32,1},idxs)

    return idxs
end

##
@everywhere function samplesurfdata(data::IndexedTable,nsample::Number=10000)
    gd = groupby(length,data,:period)
    idx_sum = 1.0 ./ select(gd,:length)
    gd = transform(gd,:idx_sum=>idx_sum)
    data = join(data,gd,lkey=:period,rkey=:period)
    dataweights = select(data,:idx_sum)
    data = transform(data,:didx=>1:length(data))
    dataidx = sample(1:length(data), Weights(dataweights), 
                  nsample,replace=false,ordered=true)
    data = filter(x -> x.didx in dataidx,data)
    #stlat = select(data,:stlat)
    #id = findall(x->x==-90.0f0,stlat)
    #stlat[id] .= -89.99f0
    #stlon = select(data,:stlon)
    #id = findall(x->x==0.0f0,stlon)
    #stlon[id] .= 0.01f0
    return data
end

##
@everywhere function geteventsweight(events::IndexedTable,
                                    nevents::Number=100)::Array{Int32,1}
    eventsloc = hcat(select(events,:evlat),
                    select(events,:evlon),select(events,:evdep))
    #eventsloc = select(events,(:evlat,:evlon,:evdep))
    idxs = geteventidx(eventsloc)
    events = transform(events,:cellidx => idxs)
    gd = groupby(length,events,:cellidx)
    idx_sum = 1.0 ./ select(gd,:length)
    gd = transform(gd,:idx_sum=>idx_sum)
    events = join(events,gd,lkey=:cellidx,rkey=:cellidx)
    eventweights = select(events,:idx_sum)
    eventid = select(events,:eventid)
    eventsused = sample(eventid, Weights(eventweights), 
                        nevents,replace=false,ordered=true)
    return eventsused 
end

##
#main function

function main()
    nthreal = 400
    nrealizations = 6 
    factor = 3.0
    phases = [["P","p","Pdiff"],["pP"],["S","s","Sdiff"]]
    jdata = load("../iscehbdata/allbodydata")
    
    ##
    events = select(jdata,(:evlat,:evlon,:evdep,:eventid))
    events = table(unique!(rows(events)))
    #nevents = 9000
    nevents = 10
    ncells = 10000
    ndatap = 200_000
    ndatas_frac = 0.95
    @sync @distributed for iter in nthreal:nthreal+nrealizations-1
    #for iter in nthreal:nthreal+nrealizations-1
        eventsusedlist = geteventsweight(events,nevents)
        @info "finish event sampling $(length(eventsusedlist))"
        jdatasub = filter(x -> x.eventid in eventsusedlist,jdata)
        eventid = select(jdatasub,:eventid)
        newidx = indexin(eventid,unique(eventid))
        newidx = convert(Array{Int32,1},newidx)
        jdatasub = transform(jdatasub,:eventidnew=>newidx)

        #sample P and S data
        jdatasubp = filter(x -> x.iss == 0, jdatasub)
        ndatap_bs = length(jdatasubp)
        nsample = min(ndatap_bs,ndatap)
        jdatasubp = jdatasubp[unique(sample(1:ndatap_bs,nsample,
                            replace=false,ordered=true))]
        dres = select(jdatasubp,:dres)
        @info "pdata before $(length(dres))"
        #q25 = factor * percentile(dres,25)
        #q75 = factor * percentile(dres,75)
        q25 = -10.0
        q75 = 10.0
        jdatasubp = filter(x -> (x.dres < q75) && (x.dres > q25),jdatasubp)
        @info "pdata $(length(jdatasubp))"

        jdatasubs = filter(x -> x.iss == 1, jdatasub)
        ndatas_bs = length(jdatasubs)
        nsample = ceil(ndatas_bs * ndatas_frac)
        nsample = convert(Int32,nsample)
        jdatasubs = jdatasubs[unique(sample(1:ndatas_bs,nsample,
                            replace=false,ordered=true))]
        dres = select(jdatasubs,:dres)
        @info "sdata before $(length(dres))"
        #q25 = factor * percentile(dres,25)
        #q75 = factor * percentile(dres,75)
        @info "percentile $(q25) and $(q75)"
        jdatasubs = filter(x -> (x.dres < q75) && (x.dres > q25),jdatasubs)
        @info "sdata $(length(jdatasubs))"

        jdatasub = merge(jdatasubp,jdatasubs)
        #jdatasub = jdatasubp
        jdatasubp = nothing
        jdatasubs = nothing 

        @info "begin subspace inversion $(length(jdatasub))"
        jdatasub = reindex(jdatasub, (:iss, :evlat, :evlon,:evdep,:stlat,
                            :stlon))
        joint = 1

        if joint == 1
            @info "begin reading surface wave data"
            #periods = loadtable("../iscehbdata/periodsnew.dat",header_exists=false,colnames=["periods"])
            #dispersyn = loadtable("../iscehbdata/disper_surf.dat",header_exists=false,colnames=["disper"])
            #nperiods = length(periods)
            #lnvp = loadtable("../iscehbdata/lnvp_sfdisp.dat",
            #                   header_exists= false,
            #                   spacedelim=true,
            #                   colnames=["period"*string(ii) for ii=1:nperiods]
            #                  )
            #lnvs = loadtable("../iscehbdata/lnvs_sfdisp.dat",
            #                   header_exists= false,
            #                   spacedelim=true,
            #                   colnames=["period"*string(ii) for ii=1:nperiods]
            #                  )
            periods = vec(readdlm("../iscehbdata/periodsnew.dat"))
            dispersyn = vec(readdlm("../iscehbdata/disper_surf.dat"))
            lnvs = readdlm("../iscehbdata/lnvs_sfdisp.dat")
            lnvp = readdlm("../iscehbdata/lnvp_sfdisp.dat")
            surfdata = load("../iscehbdata/surfdata")
            surfdata = samplesurfdata(surfdata,200000)
            #@info typeof(surfdata),length(surfdata)
        end
 
        subspaceinv(jdatasub,iter,ncells,phases,joint,periods,dispersyn,lnvp,lnvs,surfdata)
    end
    return nothing
end

@time main()
