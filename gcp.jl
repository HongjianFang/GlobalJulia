using LinearAlgebra: cross,dot
function gcp(lat1::Float64,lon1::Float64,lat2::Float64,lon2::Float64,delta::Float64)::Array{Float64,2}
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

    r0 = 6371.0
    angs2use = collect(range(0.0, stop = total_arc, length = convert(Int64,ceil(total_arc*r0/delta))))
    m = length(angs2use)

    unit_r = (ones(m,1)*reshape(ur1,1,3)).*transpose(ones(3)*reshape(cos.(angs2use),1,m)) +
             (ones(m,1)*reshape(tvec,1,3)).*transpose(ones(3)*reshape(sin.(angs2use),1,m))

    lats = asin.(unit_r[:,2])
    lons = atan.(unit_r[:,1],unit_r[:,3])

    path = hcat(lats,lons)
    dist = [total_ar*r0,r0]
    dist = reshape(dist,1,2)
    return vcat(path,dist)
end



