using Flux
using MLDatasets: MNIST
using BSON
using CSV
using DataFrames
using Printf
using Images
using Random



function Discriminator(hp)
	inchannels = 1 + length(hp.classes)
	Chain(
		Conv((3, 3), inchannels => 64; stride=2, pad=SamePad()),
		x -> leakyrelu.(x, 0.2),

		Conv((3, 3), 64 => 128; stride=2, pad=SamePad()),
		x -> leakyrelu.(x, 0.2),

		GlobalMaxPool(),
		x -> reshape(x, 128, :),

		Dense(128 => 1),
	)
end


function Generator(hp) 
	inchannels = hp.latentdim + length(hp.classes)
	Chain(
		Dense(inchannels => 7*7*inchannels),
		x -> leakyrelu.(x, 0.2),

		x -> reshape(x, 7, 7, inchannels, :),

		ConvTranspose((4, 4), inchannels => 128; stride=2, pad=SamePad()),
		x -> leakyrelu.(x, 0.2),

		ConvTranspose((4, 4), 128 => 128; stride=2, pad=SamePad()),
		x -> leakyrelu.(x, 0.2),

		Conv((7, 7), 128 => 1, σ; pad=SamePad())
	)
end


function withlabel(image, label)
	labelchannels = repeat(reshape(label, 1, 1, size(label)...), size(image)[1:2]...)
	cat(image, labelchannels, dims=3)
end

function trainstep(session, batch)
	D = session.discriminator
	G = session.generator
	opt = session.opt
	hp = session.hyperparameters

	real, labels = batch

	batchsize = size(batch)[end]
	fake = G(vcat(labels, randn(Float32, hp.latentdim, batchsize)))
	fakelabelled, reallabelled = withlabel(fake, labels), withlabel(real, labels)

	# update discriminator
	Dθ = Flux.params(D)
	Dloss, D∇ = Flux.withgradient(Dθ) do
		sum(D(fakelabelled) - D(reallabelled))
	end
	Flux.update!(opt, Dθ, D∇)

	# update generator
	Gθ = Flux.params(G)
	z = vcat(labels, randn(Float32, hp.latentdim, hp.batchsize))
	Gloss, G∇ = Flux.withgradient(Gθ) do
		-sum(D(withlabel(G(z), Float32.(labels))))
		# need to cast to Float32... bug? (https://discourse.julialang.org/t/help-me-pin-this-bug-in-flux/81901)
	end
	Flux.update!(opt, Gθ, G∇)

	push!(session.history, (Dloss, Gloss))
end




function train(session; printout_period=5)

	trainfile = "TRAINING ☢️"
	open(trainfile, "w") do file
        write(file, "Delete this file to stop training.")
    end

	for batch in session.batches
		trainstep(session, batch)

		isfile(trainfile) || break

		n = nrow(session.history)

		if iszero(n % printout_period)
			@printf "batch %5d | loss = %.5f  | G loss = %.5f\n" n session.history[end,:]...
		end

		if iszero(n % session.save_period)
			savesession(session)
		end
	end

	rm(trainfile, force=true)

end




Base.@kwdef struct Hyperparameters
	imagesize = (28, 28)
	classes = 0:9
	batchsize = 64
	latentdim = 100
end


function loadbatches(hp)
	(; features, targets) = MNIST(Float32)
	images = reshape(features, hp.imagesize..., 1, :)
	labels = Flux.onehotbatch(targets, 0:9)
	Flux.DataLoader((; images, labels); hp.batchsize, shuffle=true)
end


Base.@kwdef mutable struct TrainingSession
	path
	hyperparameters = Hyperparameters()
	discriminator = Discriminator(hyperparameters)
	generator = Generator(hyperparameters)
	opt = ADAM(0.003)
	save_period = 50
	history = DataFrame("discriminator loss"=>Float64[], "generator loss"=>Float64[])
	batches = loadbatches(hyperparameters)
end



function newsession(; path="session/", hyperparameters=Hyperparameters())
	mkpath(path)
	for sub in ["models/", "previews/"]
		mkpath(joinpath(path, sub))
	end
	open(joinpath(path, "hyperparameters.jl"), "w") do file
		write(file, repr(hyperparameters))
	end
	TrainingSession(; path, hyperparameters)
end

function loadsession(path="session/")
	hp = include(joinpath(path, "hyperparameters.jl"))
	latest = last(readdir(joinpath(path, "models/"); sort=true, join=true))
	BSON.@load latest D G
	history = CSV.read(joinpath(path, "train-history.csv"), DataFrame)
	TrainingSession(;
		path,
		hyperparameters=hp,
		discriminator=D,
		generator=G,
		history
	)
end

function savesession(session)
	D = session.discriminator
	G = session.generator
	path = session.path

	CSV.write(joinpath(path, "train-history.csv"), session.history)
	n = nrow(session.history)

	modelpath = joinpath(path, "models", @sprintf("batch-%05d.bson", n))
	BSON.@save modelpath D G

	previewpath = joinpath(path, "previews", @sprintf("batch-%05d.png", n))
	save(previewpath, preview_image(session))

	run(`cp $modelpath $(joinpath(path, "model-latest.bson"))`)
	run(`cp $previewpath $(joinpath(path, "preview-latest.png"))`)

end

function preview_image(session)
	fixed = MersenneTwister(0)

	hp = session.hyperparameters

	# preview a set of digits from fixed latent noise and a set from random noise
	z = hcat(
		vcat(Flux.onehotbatch(0:9, 0:9), randn(fixed, Float32, hp.latentdim, 10)),
		vcat(Flux.onehotbatch(0:9, 0:9), randn(Float32, hp.latentdim, 10)),
	)
	images = session.generator(z)

	Gray.(hvcat(5, eachslice(images[:,:,1,:], dims=3)...))
end