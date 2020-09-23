using Gen
using Gen3DRotations
import Rotations
import Random
using Plots

###########################################
# plots of distributions on 3D directions #
###########################################

function show_uniform_3d_direction()
    p = plot(aspect_ratio=:equal, title="uniform")
    for i in 1:1000
        x = uniform_3d_direction()
        scatter!(p, [x.v[1]], [x.v[2]], [x.v[3]], w=1, label=nothing, color="red",
            markersize=5, markerstrokealpha=0.0, alpha=0.1)
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    zlims!(-1.0, 1.0)
    return p
end

function show_vmf_3d_direction(mu, k)
    p = plot([0.0, mu.v[1]], [0.0, mu.v[2]], [0.0, mu.v[3]], w=3, label=nothing, color="blue", aspect_ratio=:equal, title="VMF(mu=$(mu.v), k=$k)")
    for i in 1:1000
        x = vmf_3d_direction(mu, k)
        scatter!(p, [x.v[1]], [x.v[2]], [x.v[3]], w=1, label=nothing, color="red",
            markersize=5, markerstrokealpha=0.0, alpha=0.1)
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    zlims!(-1.0, 1.0)
    return p
end

function show_3d_direction_dists()
    println("generating plots for 3D directions...")
    mu = UnitVector3(1.0, 0.0, 0.0)
    p1 = show_vmf_3d_direction(mu, 0.1)
    p2 = show_vmf_3d_direction(mu, 1.0)
    p3 = show_vmf_3d_direction(mu, 10.0)
    p4 = show_vmf_3d_direction(mu, 100.0)
    mu = UnitVector3(0.0, -1.0, 0.0)
    p5 = show_vmf_3d_direction(mu, 0.1)
    p6 = show_vmf_3d_direction(mu, 1.0)
    p7 = show_vmf_3d_direction(mu, 10.0)
    p8 = show_vmf_3d_direction(mu, 100.0)
    plot(
        show_uniform_3d_direction(), p1, p2, p3, p4,
        show_uniform_3d_direction(), p5, p6, p7, p8, layout=(2, 5), size=(2000, 800))
    savefig("vmf_3d_direction.png")
end

#show_3d_direction_dists()

##########################################
# plots of distributions on 3D rotations #
##########################################

function scatter_rot3!(p, rot::Rot3)
    x_axis = rot[:,1]
    y_axis = rot[:,2]
    z_axis = rot[:,3]
    scatter!(p, [x_axis[1]], [x_axis[2]], [x_axis[3]], w=1, label=nothing, color="red",
        markersize=5, markerstrokealpha=0.0, alpha=0.1)
    scatter!(p, [y_axis[1]], [y_axis[2]], [y_axis[3]], w=1, label=nothing, color="green",
        markersize=5, markerstrokealpha=0.0, alpha=0.1)
    scatter!(p, [z_axis[1]], [z_axis[2]], [z_axis[3]], w=1, label=nothing, color="blue",
        markersize=5, markerstrokealpha=0.0, alpha=0.1)
end

function plot_rot3!(p, rot::Rot3; alpha=1.0, w=5)
    x_axis = rot[:,1]
    y_axis = rot[:,2]
    z_axis = rot[:,3]
    plot!(p, [0.0, x_axis[1]], [0.0, x_axis[2]], [0.0, x_axis[3]], label=nothing, color="red", w=w, alpha=alpha)
    plot!(p, [0.0, y_axis[1]], [0.0, y_axis[2]], [0.0, y_axis[3]], label=nothing, color="green", w=w, alpha=alpha)
    plot!(p, [0.0, z_axis[1]], [0.0, z_axis[2]], [0.0, z_axis[3]], label=nothing, color="blue", w=w, alpha=alpha)
end

function show_uniform_3d_rotation()
    p = plot(aspect_ratio=:equal, title="uniform")
    for i in 1:250
        rot = uniform_rot3()
        scatter_rot3!(p, rot)
        if i == 1
            plot_rot3!(p, rot; alpha=0.5, w=2)
        end
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    zlims!(-1.0, 1.0)
    return p
end

function show_vmf_3d_rotation(mu, k)
    p = plot(aspect_ratio=:equal, title="VMF(mu=I, k=$k)")
    plot_rot3!(p, mu)
    for i in 1:250
        rot = vmf_rot3(mu, k)
        scatter_rot3!(p, rot)
        if i == 1
            plot_rot3!(p, rot; alpha=0.5, w=2)
        end
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    zlims!(-1.0, 1.0)
    return p
end

function show_uniform_vmf_3d_rotation(mu, k, prob_outlier)
    p = plot(aspect_ratio=:equal, title="UniformVMF(mu=I, k=$k, prob_outlier=$prob_outlier)")
    plot_rot3!(p, mu)
    for i in 1:250
        rot = uniform_vmf_rot3(mu, k, prob_outlier) 
        scatter_rot3!(p, rot)
        if i == 1
            plot_rot3!(p, rot; alpha=0.5, w=2)
        end
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    zlims!(-1.0, 1.0)
    return p
end

function show_3d_rotation_dists()
    println("generating plots for 3D rotations...")
    mu = Rot3(1.0, 0.0, 0.0, 0.0)
    p1 = show_vmf_3d_rotation(mu, 0.1)
    p2 = show_vmf_3d_rotation(mu, 1.0)
    p3 = show_vmf_3d_rotation(mu, 10.0)
    p4 = show_vmf_3d_rotation(mu, 100.0)
    p5 = show_vmf_3d_rotation(mu, 1000.0)
    p6 = show_uniform_vmf_3d_rotation(mu, 1000.0, 0.5)
    plot(
        p1, p2, p3, p4, p5,
        layout=(1, 5), size=(2000, 400))
    savefig("vmf_3d_rotation_1.png")
    plot(
        show_uniform_3d_rotation(), p6,
        layout=(1, 2), size=(1000, 400))
    savefig("vmf_3d_rotation_2.png")
end

show_3d_rotation_dists()

#############################################
# plots of distributions on plane rotations #
#############################################

function scatter_rot2!(p, rot::Rot2)
    rotmat = Rotations.RotMatrix{2}(rot)
    x_axis = rotmat[:,1]
    y_axis = rotmat[:,2]
    scatter!(p, [x_axis[1]], [x_axis[2]], w=1, label=nothing, color="red",
        markersize=5, markerstrokealpha=0.0, alpha=0.1)
    scatter!(p, [y_axis[1]], [y_axis[2]], w=1, label=nothing, color="green",
        markersize=5, markerstrokealpha=0.0, alpha=0.1)
end

function plot_rot2!(p, rot::Rot2; alpha=1.0, w=5)
    x_axis = rot[:,1]
    y_axis = rot[:,2]
    plot!(p, [0.0, x_axis[1]], [0.0, x_axis[2]], label=nothing, color="red", w=w, alpha=alpha)
    plot!(p, [0.0, y_axis[1]], [0.0, y_axis[2]], label=nothing, color="green", w=w, alpha=alpha)
end

function show_uniform_2d_rotation()
    p = plot(aspect_ratio=:equal, title="uniform")
    for i in 1:250
        rot = uniform_rot2()
        scatter_rot2!(p, rot)
        if i == 1
            plot_rot2!(p, rot; alpha=0.5, w=2)
        end
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    return p
end

function show_vmf_2d_rotation(mu, k)
    p = plot(aspect_ratio=:equal, title="VMF(mu=I, k=$k)")
    plot_rot2!(p, mu)
    for i in 1:250
        rot = vmf_rot2(mu, k)
        scatter_rot2!(p, rot)
        if i == 1
            plot_rot2!(p, rot; alpha=0.5, w=2)
        end
    end
    xlims!(-1.0, 1.0)
    ylims!(-1.0, 1.0)
    return p
end

function show_2d_rotation_dists()
    println("generating plots for 2D rotations...")
    mu = Rotations.RotMatrix{2}(0.0)
    @time p1 = show_vmf_2d_rotation(mu, 0.1)
    @time p2 = show_vmf_2d_rotation(mu, 1.0)
    @time p3 = show_vmf_2d_rotation(mu, 10.0)
    @time p4 = show_vmf_2d_rotation(mu, 100.0)
    @time p5 = show_vmf_2d_rotation(mu, 1000.0)
    @time plot(
        show_uniform_2d_rotation(), p1, p2, p3, p4, p5,
        layout=(1, 6), size=(2400, 400))
    @time savefig("vmf_2d_rotation.png")
end

#show_2d_rotation_dists()
