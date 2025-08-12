### A Pluto.jl notebook ###
# v0.20.15

using Markdown
using InteractiveUtils

# ╔═╡ a6fe44fb-c400-4207-b8c7-d243f735b9e7
### A Pluto.jl notebook ###
begin
	using Pkg
	Pkg.add("ImplicitAD")
	Pkg.add("ReverseDiff")
	Pkg.add("Zygote")
	Pkg.add("NLsolve")
	Pkg.add("ChainRulesCore")
	
	Pkg.add(url="https://github.com/symbiotic-engineering/MarineHydro.jl")
	Pkg.add("PyCall")
	Pkg.add("FiniteDifferences")
	using ImplicitAD
	using Zygote
	using ReverseDiff
	using MarineHydro
	using NLsolve
	using ChainRulesCore, LinearAlgebra, FiniteDifferences

	#for meshing related , call python package -Capytaine
	
	using PyCall
	# ENV["PYTHON"] = "/path/to/your/python" #python environment where capytaine is installed.
	# using Pkg
	# Pkg.build("PyCall")

end


# ╔═╡ dea1bff4-7eb6-4204-acbf-c0140da4230d
begin
	using ForwardDiff
	g(x) = sin(x) 
	x = 1.5
	println("g($x) = ", g(x))
	println("g'($x) = ", ForwardDiff.derivative(g, x))
	
end


# ╔═╡ f0f2ab01-a052-4015-ade1-367f06473a04
md"## 🔧 Introduction to Automatic Differentiation in Julia  
### 🌀 MarineHydro.jl — A Differentiable BEM Solver for Marine Hydrodynamics. 


- 🌊 Focused on **marine hydrodynamics** problems via Boundary Element Methods (BEM).
- 🧑‍💻 **Open Source** and welcoming contributors, testers, and curious users.
- 🚀 If you're excited by **differentiable physics**, **Julia**, or **offshore systems**, join us!


⚠️ **Experimental Software**: This package is in early stages — bugs are expected!"


# ╔═╡ d2f079da-6f93-456a-b918-54d7ab92b703
md"### Intro to automatic differentiation "

# ╔═╡ e655ff4e-5c5a-4895-a76c-e9b888a24fd1
md"### Forward mode differentiation like finite difference but exact and no step size business"

# ╔═╡ edb29ea8-2857-4e8c-b4be-6db510d684f7
begin
function solve_system(x)
    # Simple "solve": just compute y from x 
    y1 = x[1]^2 + sin(x[2])
    y2 = exp(x[3]) + x[4]^3
    return [y1, y2]
end

# needs a system solve
function my_program(x)
    y = solve_system(x)
    return y[1] + x[1] + y[2] + x[2]
end

x1 = [1.0, 2.0, 3.0, 4.0]

# Compute gradient using ForwardDiff.gradient
grad = ForwardDiff.gradient(my_program, x1)

println("Gradient = ", grad)
end

# ╔═╡ dbb6b39c-c4e1-481a-ad9c-f68fbcccc244
begin
	# Compute gradient using Zygote.gradient
	gradz = Zygote.gradient(my_program, x1)[1]
	println("Gradient = ", gradz)
end

# ╔═╡ da64e8f3-0403-44f8-a494-f87da48e2d57
md"### Differentiation with iterative residual solve (system solve) needs more thought
- AD engine will try to differentiate each line of the iterative algorithm but we only care that the derivative be accurate at the solution of the algorithm 
- Differentiate at the solution, use any solver for system/equation solve
"


# ╔═╡ 3919f706-fce7-4174-9e47-91e12b8b2c8c
begin
	#example from implicitAD.jl with two implicit equations
	function residual!(r, y, x, p)
	    r[1] = (y[1] + x[1])*(y[2]^3-x[2])+x[3]
	    r[2] = sin(y[2]*exp(y[1])-1)*x[4]
	end
	
	function solve(x, p)
		println("Solving the equations")
	    rwrap(r, y) = residual!(r, y, x, p)  # closure using some of the input variables within x just as an example
	    res = nlsolve(rwrap, [0.1; 1.2], autodiff=:forward)
	    return res.zero
	end
end

# ╔═╡ 37728131-01c1-4e3b-8f68-9c8cbf78d100

function modprogram(x)
    z = 2.0*x
    w = z + x.^2
    y = implicit(solve, residual!, w)
    return y[1] .+ w*y[2]
end

# ╔═╡ d6002b38-0430-4b32-ad35-fede308e5bb2
begin
	modprogram(x1)
end

# ╔═╡ b3daf591-e534-41f4-ad69-7616abb1abff
J1 = ForwardDiff.jacobian(modprogram, x1)

# ╔═╡ 33df4a52-826f-4d3a-b71d-2ac6b0af6644
md" Reverse Mode Differentiation or Adjoint mode
"

# ╔═╡ 0ddfea0a-2b19-4370-96ec-8127eebc7215
begin
	J2 = Zygote.jacobian(modprogram, x1)[1] #returns tuple so access the (jacobian,)
	println("max abs difference = ", maximum(abs.(J1 - J2)))
end

# ╔═╡ df8bb2de-0362-4c8a-901e-42de5a4a6f5c
md"""
### Automatic Differentiation in `MarineHydro.jl`

- The current implementation is **research-focused**, and the API is still evolving.
- It is designed to work seamlessly with **automatic differentiation (AD) engines**.
- For **mesh and geometry sensitivity**, we currently use **finite differences** with respect to mesh size parameters.
- In general, AD engines operate by **decomposing computations into known differentiable elementary operations**.
- For mesh-related computations, we manually supply **both the function and its derivative**, so the AD engine can simply reuse them instead of relying on symbolic rules or finite differences. 

> 📝 To do : to develop a **differentiable meshing/geometry **, enabling native support for differentiating mesh generation and deformation steps. For differentiability with respect to mesh dimension, use https://github.com/symbiotic-engineering/MarineHydro.jl/paper/MeshGradients_singlebody.jl. 

##### Here, only with respect to omega shown
"""


# ╔═╡ b91b0dc0-da09-43e2-969c-be681e092c4f


# ╔═╡ f9f9b4f2-62c9-11f0-3438-27cf7886b3aa

begin
# import your capytaine mesh
cpt = pyimport("capytaine")
radius = 1.0 #fixed
resolution = (20, 20)
cptmesh = cpt.mesh_sphere(name="sphere", radius=radius, center=(0, 0, 0), resolution=resolution) 
cptmesh.keep_immersed_part(inplace=true)

# declare it Julia mesh
mesh = Mesh(cptmesh)  
ω = 1.03
ζ = [0,0,1] # HEAVE: will be more verbose in future iteration. define it again even if defined in Capytaine.
	
#MarineHydro experimental API 
	
F = DiffractionForce(mesh,ω,ζ)
A,B = calculate_radiation_forces(mesh,ζ,ω)
end


# ╔═╡ e004d41e-c4f1-4002-99df-1036f9a28dd4
begin 
function check_added_mass(ω,mesh,dof)
        A = calculate_radiation_forces(mesh,dof,ω)[1]
        return A
    end

Am(w) = check_added_mass(w, mesh,ζ)
A_w_grad = ForwardDiff.derivative(Am,ω)
print(A_w_grad)
end

# ╔═╡ 14defa7e-8857-4fcd-8177-4e82bdd6b261
#checking accuracy with FiniteDifferences
# Central difference with order 5
begin
fd_grad1 = FiniteDifferences.central_fdm(5, 1)(Am, ω)
print(fd_grad1)
end


# ╔═╡ 0f9ff588-fe85-410f-8666-e5c3ba725507
md"#### Exercise: Reverse mode using Zygote
- Reverse mode is useful for design optimization.
- Zygote / ReverseDiff / Enzyme support reverse-mode differentiation if your code allows gradient propagation. 

Our paper implements reverse mode through mesh for design optimization. Setting up meshing is difficult with Python and Julia currently and hence we need to implement meshing next in Julia"


# ╔═╡ deb89ef6-77b1-4b40-8ddb-214907376cb7


# ╔═╡ a14e88eb-3047-4501-9142-cccafc4150c7
md" ## There could be BIG BUGS in this research software at this point. We are looking for contributors and feedback on it

### To do:
1) Ireggular frequency removal
2) GPU, Distributed Computing etc
3) Software API 
4) Speeding up automatic differentiation by providing analytical gradient to AD engine. Less work for AD engine!!
5) AD usecases
6) Mesh in Geometry
"

# ╔═╡ Cell order:
# ╟─f0f2ab01-a052-4015-ade1-367f06473a04
# ╟─a6fe44fb-c400-4207-b8c7-d243f735b9e7
# ╟─d2f079da-6f93-456a-b918-54d7ab92b703
# ╠═dea1bff4-7eb6-4204-acbf-c0140da4230d
# ╟─e655ff4e-5c5a-4895-a76c-e9b888a24fd1
# ╠═edb29ea8-2857-4e8c-b4be-6db510d684f7
# ╠═dbb6b39c-c4e1-481a-ad9c-f68fbcccc244
# ╠═da64e8f3-0403-44f8-a494-f87da48e2d57
# ╠═3919f706-fce7-4174-9e47-91e12b8b2c8c
# ╠═37728131-01c1-4e3b-8f68-9c8cbf78d100
# ╠═d6002b38-0430-4b32-ad35-fede308e5bb2
# ╠═b3daf591-e534-41f4-ad69-7616abb1abff
# ╟─33df4a52-826f-4d3a-b71d-2ac6b0af6644
# ╠═0ddfea0a-2b19-4370-96ec-8127eebc7215
# ╟─df8bb2de-0362-4c8a-901e-42de5a4a6f5c
# ╠═b91b0dc0-da09-43e2-969c-be681e092c4f
# ╠═f9f9b4f2-62c9-11f0-3438-27cf7886b3aa
# ╠═e004d41e-c4f1-4002-99df-1036f9a28dd4
# ╠═14defa7e-8857-4fcd-8177-4e82bdd6b261
# ╠═0f9ff588-fe85-410f-8666-e5c3ba725507
# ╠═deb89ef6-77b1-4b40-8ddb-214907376cb7
# ╠═a14e88eb-3047-4501-9142-cccafc4150c7
