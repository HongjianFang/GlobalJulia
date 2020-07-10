__precompile__()

using Distributed
@everywhere import PyCall 
@everywhere using StatsBase: sample,Weights,percentile
@everywhere using IterativeSolvers: lsmr
@everywhere using SparseArrays: sparse
@everywhere using NearestNeighbors: KDTree,knn
@everywhere using HDF5#: h5read,h5write,h5open
@everywhere using JuliaDB
@everywhere using LinearAlgebra

@everywhere taup = PyCall.pyimport("obspy.taup")
@everywhere model = taup.TauPyModel(model="ak135")
@everywhere get_ray_paths_geo = model.get_ray_paths_geo

@everywhere const HVR = 2.0
@everywhere const EARTH_CMB = 3481.0
@everywhere const EARTH_RADIUS = 6371.0

@everywhere function wrap_get_ray_paths_geo(evdep::Float32,evlat::Float32,evlon::Float32,stlat::Float32,stlon::Float32,phase_list::Array{String,1},sampleds::Float32)
    arr = get_ray_paths_geo(evdep,evlat,evlon,stlat,stlon,phase_list,true,sampleds)
    rayparam = 0.0
    raytakeoffangle = 0.0
    ifray = true
    #raypts = Float32[]
    if length(arr)>0
        lon = get(arr[1].path,"lon")   
        lon = deg2rad.(lon)
        lat = get(arr[1].path,"lat")   
        lat = pi/2.0 .- deg2rad.(lat)
        dep = get(arr[1].path,"depth")   
        rad = EARTH_RADIUS .- dep .* HVR
        raypts = hcat(rad,lat,lon)
        raypts = convert(Array{Float32,2},raypts)
        rayparam = arr[1].ray_param
        raytakeoffangle = arr[1].takeoff_angle
    else 
        ifray = false
    end
    return (ifray,rayparam,raytakeoffangle,raypts)
end

@everywhere function subspaceinv(datasub::IndexedTable,iter::Number,ncells::Number,phases::Array{Array{String,1},1})
    mindist = convert(Float32,2.0)
    nlat = 512
    nlon = 1024
    nrad = 128
    factor = 3.0
    k = 1
    
    cellsph = generate_vcells(ncells)
    cellxyz = sph2xyz(cellsph)
    kdtree = KDTree(cellxyz;leafsize=10)
    
    row = Int32[]
    col = Int32[]
    values = Float32[]
    b = Float32[]
    dres = select(datasub,:dres)
    q25 = factor * percentile(dres,25)
    q75 = factor * percentile(dres,75)
    datasub = filter(x -> (x.dres < q75) && (x.dres > q25),datasub)
    dataidx = 0
    ndata = length(datasub)
    println("start sensitivity mastrix $(ndata)");flush(stdout)
    @info "start sensitivity mastrix $(ndata)"

    evlatall = select(datasub,:evlat)
    evlonall = select(datasub,:evlon)
    evdepall = select(datasub,:evdep)
    stlatall = select(datasub,:stlat)
    stlonall = select(datasub,:stlon)
    azimall = select(datasub,:azim)
    eventidnewall = select(datasub,:eventidnew)
    phaseall = select(datasub,:phase)
    issall = select(datasub,:iss)
    datasuball = select(datasub,:dres)

    for ii in 1:ndata
        evlat = evlatall[ii]
        evlon = evlonall[ii]
        evdep = evdepall[ii]
        stlat = stlatall[ii]
        stlon = stlonall[ii]
        azimulth = azimall[ii]
        srcidx =  eventidnewall[ii]
        phaseno = phaseall[ii]
        iss =     issall[ii]
        bres = datasuball[ii]
        phase = phases[phaseno]
        ray = wrap_get_ray_paths_geo(evdep, evlat, evlon, stlat, stlon, phase, mindist)
        ifray = ray[1]
        if !ifray
            continue
        end
        rayparam = ray[2]
        raytakeoffangle = ray[3]
        raysph = ray[4]#convert(Array{Float32,2},ray[4])

        dataidx += 1
        rayxyz = sph2xyz(raysph)
        idxs, _ = knn(kdtree, rayxyz, k, false)
        idxs = [x[1] for x in idxs]
        rowray = zeros(Float32,ncells)
        for id in idxs
            rowray[id] += mindist
        end
        colid = findall(x->x>0,rowray)# .+ iss*ncells
        nonzeros = rowray[colid]
        colid = colid .+ iss*ncells
        rowid = ones(Int32,size(colid)) .* dataidx
        append!(col,colid)
        append!(row,rowid)
        append!(values,nonzeros)
        
        #relocation
        colid = zeros(Int32,4)
        rowid = zeros(Int32,4)
        nonzeros = zeros(Float32,4)
        deltar = 10.0
        colid[1] = 2*ncells+srcidx*4+1
        rowid[1] = dataidx
        nonzeros[1] = rayparam/tan(raytakeoffangle)/EARTH_RADIUS*deltar
        
        colid[2] = 2*ncells+srcidx*4+2
        rowid[2] = dataidx
        nonzeros[2] = -rayparam*cos(azimulth)/EARTH_RADIUS*deltar
        colid[3] = 2*ncells+srcidx*4+3
        rowid[3] = dataidx
        nonzeros[3] = rayparam*sin(azimulth)*cos(deg2rad(evlat))/EARTH_RADIUS*deltar
        colid[4] = 2*ncells+srcidx*4+4
        rowid[4] = dataidx
        nonzeros[4] = 1.0
        
        append!(col,colid)
        append!(row,rowid)
        append!(values,nonzeros)
        append!(b,bres)
    end

    println("Finishing ray tracing with nonzeros: $(length(values))");flush(stdout)
    G = sparse(row,col,values)
    
    ncol = size(G,2)
    cnorm = zeros(Float32,ncol)
    for icol = 1 : ncol
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


    row = 0
    col = 0
    values = 0
    
    damp = 1.0
    atol = 1e-4
    btol = 1e-6
    conlim = 100
    maxiter = 100
    x = lsmr(G,b,λ=damp, atol = atol, btol = btol,log = true)
    println(x[2])

    x = x[1]
    x ./= cnorm
    x = convert(Array{Float32,1},x)
    xall = zeros(Float32,ncol)
    xall[colid] = x
    xp = xall[1:ncells]
    xs = xall[ncells+1:2*ncells]
    xall = 0.0

    print("begin projection matrix");flush(stdout)
    @info "begin projection matrix"
    dlat = pi/nlat
    dlon = 2*pi/nlon
    drad = (EARTH_RADIUS-EARTH_CMB)/nrad
    lat = -(pi-dlat)/2.0:dlat:(pi-dlat)/2.0
    lat = collect(pi/2.0 .- lat)
    lat = convert(Array{Float32},lat)
    lon = dlon/2.0:dlon:2*pi
    lon = collect(lon)
    lon = convert(Array{Float32},lon)
    rad = EARTH_RADIUS .- HVR.*(drad/2.0:drad:nrad*drad-drad/2.0)
    rad = collect(rad)
    rad = convert(Array{Float32},rad)
    npara = nlat*nlon*nrad
    #latall = reshape([xj for xj in lat for yj in lon for zj in rad], npara)
    #lonall = reshape([yj for xj in lat for yj in lon for zj in rad], npara)
    #radall = reshape([zj for xj in lat for yj in lon for zj in rad], npara)

    latall = zeros(Float32,npara)
    lonall = zeros(Float32,npara)
    radall = zeros(Float32,npara)

    idx = 0
    for xj in lat
        for yj in lon
            for zj in rad
                idx += 1
                latall[idx] = xj
                lonall[idx] = yj
                radall[idx] = zj
            end
        end
    end
    
    gridsph = hcat(radall,latall,lonall)
    #gridsph = convert(Array{Float32,2},gridsph)
    gridxyz = sph2xyz(gridsph)
    radall = 0 
    latall = 0
    lonall = 0
    gridsph = 0
    
    println("Finishing projection matrix")
    k = 1
    colgp, _ = knn(kdtree, gridxyz, k, false)
    gridxyz = 0
    colgp = [x[1] for x in colgp]
    colgp = convert(Array{Int32,1},colgp)
    rowgp = convert(Array{Int32,1},collect(1:npara))
    valuegp = ones(Float32,npara)
    Gp = sparse(rowgp,colgp,valuegp,npara,ncells)

    rowgp = 0
    colgp = 0
    valuegp = 0 
 
    vp = Gp*xp
    vs = Gp*xs
    vp = convert(Array{Float32},vp)
    vs = convert(Array{Float32},vs)
    h5open("juliadata/vp$(iter).h5","w") do file
        write(file,"vp",vp)
    end
    h5open("juliadata/vs$(iter).h5","w") do file
        write(file,"vs",vs)
    end
    return nothing#vp[1]
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
    cellrad =  stretchradialcmb .* rand(ncells) .+ EARTH_RADIUS .- stretchradialcmb
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
@everywhere function geo2sph(geopts::Array{Float32,2})::Array{Float32,2}
    lat = geopts[:,1]
    lon = geopts[:,2]
    depth = geopts[:,3]
    npoints = length(lat)
    sph = zeros(Float32,npoints,3)
    sph[:,1] = -depth .+ EARTH_RADIUS 
    sph[:,2] = -deg2rad.(lat) .+ pi/2.0  
    sph[:,3] = deg2rad.(lon)
    return sph
end
##

##
#"""
#transform spherical coordinates to Cartician
#Args
#rad,phi,theta
#"""
@everywhere function sph2xyz(sph::Array{Float32,2})::Array{Float32,2}
    npoints,_ = size(sph)
    #rad = sph[:,1]
    #phi = sph[:,2]
    #theta = sph[:,3]
    xyz = zeros(Float32,npoints,3)
    xyz[:,1] = sph[:,1] .* sin.(sph[:,2]) .* cos.(sph[:,3])
    xyz[:,2] = sph[:,1] .* sin.(sph[:,2]) .* sin.(sph[:,3])
    xyz[:,3] = sph[:,1] .* cos.(sph[:,2]) 
    return transpose(xyz)
end
##

##
#"""
#sample events based on their distribution
#"""
@everywhere function geteventidx(eventsloc::Array{Float32,2},eventcell::Number = 5000)::Array{Int32,1}

    cellphi = acos.(rand(eventcell).*2 .- 1.0)
    celltheta = 2.0*pi*rand(eventcell)
    cellrad = EARTH_RADIUS .- 660.0 * rand(eventcell) 

    sph = hcat(cellphi,celltheta,cellrad)
    sph = convert(Array{Float32,2},sph)

    cellxyz = sph2xyz(sph)
    kdtree = KDTree(cellxyz;leafsize=10)
    sph = geo2sph(eventsloc)
    xyz = sph2xyz(sph)
    k = 1
    idxs, _ = knn(kdtree, xyz, k, false)
    idxs = [x[1] for x in idxs]
    idxs = convert(Array{Int32,1},idxs)

    return idxs
end

##
@everywhere function geteventsweight(events::IndexedTable,nevents::Number=100)::Array{Int32,1}
    eventsloc = hcat(select(events,:evlat),select(events,:evlon),select(events,:evdep))
    idxs = geteventidx(eventsloc)
    events = transform(events,:cellidx => idxs)
    gd = groupby(length,events,:cellidx)
    idx_sum = 1.0 ./ select(gd,:length)
    gd = transform(gd,:idx_sum=>idx_sum)
    events = join(events,gd,lkey=:cellidx,rkey=:cellidx)
    eventweights = select(events,:idx_sum)
    eventid = select(events,:eventid)
    eventsused = sample(eventid, Weights(eventweights),nevents)
    return eventsused 
end

##
#main function

function main()
    nrealizations = 3 
    phases = [["P","p","Pdiff"],["pP"],["S","s","Sdiff"]]
    data = h5read("../randmesh_global/jointdataset/jointdata_isc.h5","data")
    keyss = vcat(data["block0_items"],data["block1_items"])
    datasub = vcat(data["block0_values"],data["block1_values"])
    datasub = convert(Array{Float32,2},transpose(datasub))
    jdata = table(datasub[:,1],datasub[:,2],datasub[:,3],datasub[:,4],
                  datasub[:,5],datasub[:,6],datasub[:,7],datasub[:,8];
                  names=[Symbol(ikey) for ikey in keyss])
    bdata = convert(Array{Int32,1},vec(data["block2_values"]))
    jdata = transform(jdata,Symbol(data["block2_items"][1])=>bdata)
    bdata = convert(Array{Int32,1},vec(data["block3_values"]))
    jdata = transform(jdata,Symbol(data["block3_items"][1])=>bdata)
    data = 0.0
    bdata = 0.0
    datasub = 0.0
    iss = map(x -> x.phase<3 ? 0 : 1,jdata)
    jdata = transform(jdata,:iss=>iss)
    
    ##
    events = select(jdata,(:evlat,:evlon,:evdep,:eventid))
    events = table(unique!(rows(events)))
    nevents = 4000
    ncells = 20000
    ndatap = 400_000
    ndatas_frac = 0.9
    @sync @distributed for iter in 90:89+nrealizations
        eventsusedlist = geteventsweight(events,nevents)
        println("finish event sampling");flush(stdout)
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
        jdatasubp = jdatasubp[unique(sample(1:ndatap_bs,nsample,ordered=true))]

        jdatasubs = filter(x -> x.iss == 1, jdatasub)
        ndatas_bs = length(jdatasubs)
        nsample = ceil(ndatas_bs * ndatas_frac)
        nsample = convert(Int32,nsample)
        jdatasubs = jdatasubs[unique(sample(1:ndatas_bs,nsample,ordered=true))]
        jdatasub = merge(jdatasubp,jdatasubs)
        jdatasubp = 0
        jdatasubs = 0 

        println("begin subspace inversion");flush(stdout)
        @info "begin subspace inversion $(length(jdatasub))"
        subspaceinv(jdatasub,iter,ncells,phases)
    end
    return nothing
end

@time main()
