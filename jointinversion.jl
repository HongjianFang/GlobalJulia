__precompile__()
using PyCall
using StatsBase
using IterativeSolvers
using SparseArrays
using NearestNeighbors
using HDF5
using DataFrames

const HVR = 2.0
const EARTH_CMB = 3481.0
const EARTH_RADIUS = 6371.0
##

#module pytaup
#using PyCall
#export taup
#
#const taup = PyNULL()
#
#function __init__()
#        copy!(taup, pyimport_conda("obspy.taup", "obspy"))
#    end
#end

##

function subspaceinv(datasub::DataFrame,iter::Int64,ncells::Int64=20000)
    taup = pyimport("obspy.taup")
    model = taup.TauPyModel(model="ak135")
    phases = [["P","p","Pdiff"],["pP"],["S","s","Sdiff"]]
    mindist = 2.0
    nlat = 512
    nlon = 1024
    nrad = 128
    factor = 3.0
    k = 1
    
    cellsph = generate_vcells(ncells)
    cellxyz = sph2xyz(cellsph)
    kdtree = KDTree(cellxyz;leafsize=10)
    
   
    row = Int64[]
    col = Int64[]
    values = Float64[]
    b = Float64[]
    dres = datasub[!,:dres]
    q25 = factor * percentile(dres,25)
    q75 = factor * percentile(dres,75)
    datasub = datasub[(datasub.dres .< q75) .& (datasub.dres .> q25),:]
    dataidx = 0
    ndata,_ = size(datasub)
    println("start sensitivity mastrix $(ndata)")
    @info "start sensitivity mastrix $(ndata)"
    for ii in 1:ndata
        #print("The $(ii)'th meansurement\r")
        evlat = datasub[ii,:evlat]
        evlon = datasub[ii,:evlon]
        evdep = datasub[ii,:evdep]
        stlat = datasub[ii,:stlat]
        stlon = datasub[ii,:stlon]
        azimulth = datasub[ii,:azim]
        srcidx = datasub[ii,:eventidnew]
        phaseno = datasub[ii,:phase]
        iss = datasub[ii,:iss]
        phase = phases[phaseno]
        arr = model.get_ray_paths_geo(evdep, evlat, evlon, stlat, stlon, phase_list = phase, resample=true, sampleds=mindist)
        if length(arr) < 1
            continue
        end
        rayparam = arr[1].ray_param
        raytakeoffangle = arr[1].takeoff_angle
        
        dataidx += 1
        lon = get(arr[1].path,"lon")   
        lat = get(arr[1].path,"lat")   
        dep = get(arr[1].path,"depth")   
        raypts = hcat(lat,lon,dep)
        raysph = geo2sph(raypts)
        rayxyz = sph2xyz(raysph)
        idxs, _ = knn(kdtree, rayxyz, k, false)
        idxs = [x[1] for x in idxs]
        rowray = zeros(Float64,ncells)
        for id in idxs
            rowray[id] += mindist
        end
        colid = findall(x->x>0,rowray)# .+ iss*ncells
        nonzeros = rowray[colid]
        colid = colid .+ iss*ncells
        rowid = ones(Int64,size(colid)) .* dataidx
        append!(col,colid)
        append!(row,rowid)
        append!(values,nonzeros)
        
        #relocation
        colid = zeros(Int64,4)
        rowid = zeros(Int64,4)
        nonzeros = zeros(Float64,4)
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
        append!(b,datasub[ii,:dres])
    end

    #println("nonzeros: $(length(nonzeros))")
    G = sparse(row,col,values)
    row = []
    col = []
    values = []
    
    damp = 0.1
    atol = 1e-4
    btol = 1e-6
    conlim = 100
    maxiter = 100
    x = lsmr(G,b,λ=damp, atol = atol, btol = btol,log = true)

    x = x[1]
    xp = x[1:ncells]
    xs = x[ncells+1:2*ncells]

    print("begin projection matrix")
    @info "begin projection matrix"
    dlat = pi/nlat
    dlon = 2*pi/nlon
    drad = (EARTH_RADIUS-EARTH_CMB)/nrad
    lat = -(pi-dlat)/2.0:dlat:(pi-dlat)/2.0
    lat = collect(pi/2.0 .- lat)
    lon = dlon/2.0:dlon:2*pi
    lon = collect(lon)
    rad = EARTH_RADIUS .- HVR.*(drad/2.0:drad:nrad*drad-drad/2.0)
    rad = collect(rad)
    npara = nlat*nlon*nrad
    latall = reshape([xj for xj in lat for yj in lon for zj in rad], npara)
    lonall = reshape([yj for xj in lat for yj in lon for zj in rad], npara)
    radall = reshape([zj for xj in lat for yj in lon for zj in rad], npara)
    
    gridsph = hcat(radall,latall,lonall)
    gridxyz = sph2xyz(gridsph)
    radall = []
    latall = []
    lonall = []
    
    colgp, _ = knn(kdtree, gridxyz, k, false)
    colgp = [x[1] for x in colgp]
    rowgp = collect(1:npara)
    valuegp = ones(Float64,npara)
    Gp = sparse(rowgp,colgp,valuegp,npara,ncells)

    gridxyz = []
    rowgp = []
    colgp = []
    valuegp = []
 
    vp = Gp*xp
    vs = Gp*xs
    vp = convert(Array{Float32},vp)
    vs = convert(Array{Float32},vs)
    #vel = hcat(vp,vs)
    #vel = convert(Array{Float32},vel)
    #h5write(tempdir*"/vp$(iter).h5","vp",vel[:,1])
    #h5write(tempdir*"/vs$(iter).h5","vs",vel[:,2])
    h5write("juliadata/vp$(iter).h5","vp",vp)
    h5write("juliadata/vs$(iter).h5","vs",vs)
    return nothing
end


##
"""
generate vcells
"""

function generate_vcells(ncells::Int64=20000)::Array{Float64,2}
    cellphi = acos.(rand(ncells).*2 .- 1.0)# .- pi/2.0
    celltheta = 2.0*pi*rand(ncells)
    stretchradialcmb = ( EARTH_RADIUS - EARTH_CMB ) .* HVR
    cellrad =  stretchradialcmb .* rand(ncells) .+ EARTH_RADIUS .- stretchradialcmb
    cellptssph = hcat(cellrad,cellphi,celltheta)
    return cellptssph
end
##

"""
transform geo-coordinates to spherical coordinates
Args
lat,lon,dep
"""
#function geo2sph(lat::Array{Float64,1},lon::Array{Float64,1},depth::Array{Float64,1})
function geo2sph(geopts::Array{Float64,2})::Array{Float64,2}
    lat = geopts[:,1]
    lon = geopts[:,2]
    depth = geopts[:,3]
    npoints = length(lat)
    sph = zeros(Float64,npoints,3)
    sph[:,1] = -depth .+ EARTH_RADIUS 
    sph[:,2] = -deg2rad.(lat) .+ pi/2.0  
    sph[:,3] = deg2rad.(lon)
    return sph
end
##

##
"""
transform spherical coordinates to Cartician
Args
rad,phi,theta
"""
function sph2xyz(sph::Array{Float64,2})::Array{Float64,2}
    npoints,_ = size(sph)
    rad = sph[:,1]
    phi = sph[:,2]
    theta = sph[:,3]
    xyz = zeros(Float64,npoints,3)
    xyz[:,1] = rad .* sin.(phi) .* cos.(theta)
    xyz[:,2] = rad .* sin.(phi) .* sin.(theta)
    xyz[:,3] = rad .* cos.(phi) 
    return transpose(xyz)
end
##

##
"""
sample events based on their distribution
"""
function geteventidx(eventsloc::Array{Float64,2},eventcell::Int64 = 5000)::Array{Int64,1}

    cellphi = acos.(rand(eventcell).*2 .- 1.0)
    celltheta = 2.0*pi*rand(eventcell)
    cellrad = EARTH_RADIUS .- 660.0 * rand(eventcell) 

    sph = hcat(cellphi,celltheta,cellrad)

    cellxyz = sph2xyz(sph)
    kdtree = KDTree(cellxyz;leafsize=10)
    sph = geo2sph(eventsloc)
    xyz = sph2xyz(sph)
    k = 1
    idxs, _ = knn(kdtree, xyz, k, false)
    idxs = [x[1] for x in idxs]

    return idxs
end

##
function geteventsweight(events::DataFrame,nevents::Int64=100)::Array{Int64,1}
    eventsloc = hcat(events[!,:evlat],events[!,:evlon],events[!,:evdep])
    idxs = geteventidx(eventsloc)
    events[!,:cellidx] = idxs
    gd = combine(groupby(events,:cellidx),:eventid,:cellidx=>length)
    #gd = combine(gd,:cellidx => count)
    gd[!,:idx_sum] = 1.0./gd[!,:cellidx_length]
    gd = gd[!,[:eventid,:idx_sum]]
    items = gd[!,:eventid]
    weights = gd[!,:idx_sum]
    eventid = events[:,:eventid]
    eventsused = sample(eventid, Weights(weights),nevents)
    return eventsused 
end

##
#main function


nrealizations = 5
#tempdir = "juliadata"
#if ! isdir(tempdir)
#    mkdir(tempdir)
#end
data = h5read("../randmesh_global/jointdataset/jointdata_isc.h5","data")
keyss = vcat(data["block0_items"],data["block1_items"])
datasub = vcat(data["block0_values"],data["block1_values"])
jdata = DataFrame(transpose(datasub),keyss)
jdata[!,Symbol(data["block2_items"][1])] = vec(data["block2_values"])
jdata[!,Symbol(data["block3_items"][1])] = vec(data["block3_values"])
jdata[!,:iss] .= 0
jdata[jdata.phase .== 3, :iss ] .= 1

##
events = jdata[:,[:evlat,:evlon,:evdep,:eventid]]
unique!(events)
nevents = 1000
ncells = 20000

for iter in 1:nrealizations
    eventsused = copy(events)
    eventsusedlist = geteventsweight(eventsused,nevents)
    println("finish event sampling")
    @info "finish event sampling"
    jdatasub = jdata[in.(jdata.eventid,Ref(eventsusedlist)),:]
    eventid = jdatasub[!,:eventid]
    jdatasub[!,:eventidnew] = indexin(eventid,unique(eventid))
    println("begin subspace inversion")
    @info "begin subspace inversion"
    subspaceinv(jdatasub,iter,ncells)
end

