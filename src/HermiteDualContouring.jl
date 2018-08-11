module HermiteDualContouring

using RegionTrees
using ConstructiveSolidGeometry: Surface, distance_field, normal_field, unitize
using Optim
using LineSearches

import RegionTrees: needs_refinement, refine_data

export HermiteDualContour

struct DualContourRefinery{S<:Surface} <: AbstractRefinery
    surface_def::S
    atol::Float64
    rtol::Float64
    surfcellmax::Float64
end

function find_surface_intersect(v1, v2, contour::T) where {T<:Surface}
    edge_func = (α)->v1+α*(v2-v1)
    line_func = (α)->(distance_field(contour, edge_func(α))^2)
    #Because of how edge_func is parameterized, the range is always [0,1]
    result = optimize(line_func,0.0,1.0,GoldenSection())
    α_final = Optim.minimizer(result)
    p = edge_func(α_final)
    return p
end

function needs_refinement(refinery::DualContourRefinery, cell::Cell)
    minimum(cell.boundary.widths) > refinery.atol || return false

    if !isempty(cell.data.residual) && !isnan(cell.data.residual)
        if minimum(cell.boundary.widths)>refinery.surfcellmax
            return true
        elseif !isapprox(cell.data.residual, 0.0, rtol=refinery.rtol, atol=refinery.atol)
            return true
        else
            return false
        end
    else # Don't check for zero or odd normals, just break up the cell if the QEF fails. Is this robust/efficient?
        if isempty(cell.data.residual)
            # check to see if the distances from the corners could indicate a surface within the cell
            max_width = maximum(cell.boundary.widths)
            for vertex in vertices(cell)
                if abs(distance_field(refinery.surface_def,vertex)) <= max_width/sqrt(2)
                    return true #still creates a lot of fine granularity in empty space, but much better
                end
            end
            return false #line only reached if no vertices meet distance condition
        else
            if minimum(cell.boundary.widths)>refinery.surfcellmax
                return true
            else
                return false #isnan case
            end
        end
    end
end

function refine_data(refinery::DualContourRefinery, cell::Cell, indices)
    refine_data(refinery, child_boundary(cell, indices))
end

struct CellData
    edges
    p
    n
    qef_min
    residual
    function CellData(
        edges=[],
        p=[],
        n=[],
        qef_min=[],
        residual=[])
        new(edges,p,n,qef_min,residual)
    end
end

function isOutside(vert_array, c, rtol, atol)
    cat_verts = hcat(collect(vert_array)...)
    maxX = maximum(cat_verts[1,:])
    minX = minimum(cat_verts[1,:])

    maxY = maximum(cat_verts[2,:])
    minY = minimum(cat_verts[2,:])

    if !isapprox(maxX,c[1], rtol=rtol, atol=atol) && maxX<c[1]
        return true
    elseif !isapprox(minX,c[1], rtol=rtol, atol=atol) && minX>c[1]
        return true
    elseif !isapprox(maxY,c[2], rtol=rtol, atol=atol) && maxY<c[2]
        return true
    elseif !isapprox(minY,c[2], rtol=rtol, atol=atol) && minY>c[2]
        return true
    else
        return false
    end
end

function refine_data(refinery::DualContourRefinery, boundary::HyperRectangle)
    p=[]
    n=[]
    edges=[]
    vert_array=vertices(boundary) #ideally this would be examining edges not vertices
    v_order = [1,2,4,3,1] #2D for now, edges would not require the list

    #logic to check if cell is entirely in/outside of surface
    val_array = map((x)->distance_field(refinery.surface_def, x), vert_array)#for now, 2D
    sign_set = Set(sign.(val_array))
    is_on_plane = length(val_array)-countnz(val_array)>1 && -1.0 in sign_set
    is_not_on_surface = length(sign_set)==1 && !(0.0 in sign_set || -0.0 in sign_set) #what to do if all verts are on surface?
    has_single_corner = (length(sign_set)<3 && (0.0 in sign_set || -0.0 in sign_set)) #because floats are terrible
    if  !is_on_plane && (is_not_on_surface || has_single_corner)
        #check to see if the cell contains a surface or part of one within it
        return CellData()
    end

    for v_idx = 1:length(vert_array)
        current_vert = vert_array[v_order[v_idx]]
        next_vert = vert_array[v_order[v_idx+1]]

        current_val = val_array[v_order[v_idx]]
        next_val = val_array[v_order[v_idx+1]]

        current_sign = sign(current_val)
        next_sign = sign(next_val)

        sign_change = current_sign!=next_sign
        current_sign_zero = current_sign==0.0 || current_sign==-0.0
        next_sign_zero = next_sign==0.0 || next_sign == -0.0

        if sign_change || (current_sign_zero && next_sign_zero)
            if current_val==zero(current_val)
                p_local = current_vert
            elseif next_val==zero(next_val)
                continue
            else
                p_local = find_surface_intersect(current_vert, next_vert, refinery.surface_def)
            end
            n_local = normal_field(refinery.surface_def, p_local) #for now, 2D
            push!(p,p_local)
            push!(n,n_local) #for now, 2D
            push!(edges, [current_vert, next_vert])
        end
    end

    A = hcat(n...)'
    simA=similar(A,size(A)...)
    map!(x->x,simA,A) #convert out of SMatrix so that A\b works correctly
    b = dot.(p,n)
    qef_min=0#remember how to initialize as undefined
    residual=0#remember how to initialize as undefined
    if length(p)==1 #may be useful for debugging where flat surfaces are not involved
        qef_min = Inf
        residual = Inf
    else
        try
            qef_min = simA\b
            residual = distance_field(refinery.surface_def, qef_min)
        catch
            qef_min = NaN
            residual = NaN
        end
        if !isnan(residual)
            #occurs when a cell corner aligns with a surface corner, needs to be adjusted for dimensionality
            corner_idx = find(i->isapprox(i,qef_min), p)
            if !isempty(corner_idx) && length(p)>2
                deleteat!(edges, corner_idx)
                deleteat!(p, corner_idx)
                deleteat!(n, corner_idx)
            end
            if isOutside(vert_array,qef_min, refinery.rtol, refinery.atol)
                qef_min = Inf
                residual = Inf
            end
        end
    end

    return CellData(
        edges,
        p,
        n,
        qef_min,
        residual)
end

struct HermiteDualContour{C<:Cell} <: Surface
    root::C
end

function HermiteDualContour(surface_def::Surface, origin::AbstractArray,
        widths::AbstractArray,
        rtol = 1e-2,
        atol = 1e-2,
        surfcellmax = 1e-1)
    refinery = DualContourRefinery(surface_def, atol, rtol, surfcellmax)
    boundary = HyperRectangle(origin, widths)
    root = Cell(boundary, refine_data(refinery, boundary))
    adaptivesampling!(root, refinery)
    HermiteDualContour(root)
end

# end


end # module
