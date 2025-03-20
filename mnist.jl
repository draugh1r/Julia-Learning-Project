# MNIST classification with Flux.jl

using Flux, MLDatasets, Random, Statistics

Random.seed!(42)  # for reproducibility

# load dataset
train_data = MLDatasets.MNIST(:train)
test_data = MLDatasets.MNIST(:test)

# preprocess data
X_train = Float32.(reshape(train_data.features, 28*28, :)) ./ 255.0  # flatten and normalize
y_train = Flux.onehotbatch(train_data.targets .+ 1, 1:10)  # one-hot encoding
X_test = Float32.(reshape(test_data.features, 28*28, :)) ./ 255.0
y_test = Flux.onehotbatch(test_data.targets .+ 1, 1:10)

# define MLP model
model = Chain(
    Dense(28*28, 128, relu),  # input layer
    Dense(128, 64, relu),  # hidden layer
    Dense(64, 10),  # output layer
    softmax  # probability distribution
)

# loss function & optimizer
loss(x, y) = crossentropy(model(x), y)
opt = ADAM()

# train model
epochs = 5
for epoch in 1:epochs
    Flux.train!(loss, Flux.params(model), [(X_train, y_train)], opt)
    println("Epoch $epoch completed")
end

# evaluate accuracy
accuracy(x, y) = mean(argmax(model(x), dims=1) .== argmax(y, dims=1))
println("Test Accuracy: ", accuracy(X_test, y_test))
