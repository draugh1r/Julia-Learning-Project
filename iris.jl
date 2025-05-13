# K-means clustering on Iris dataset

using MLDatasets, Clustering, Statistics, Plots, DataFrames, Random

Random.seed!(42)  # for reproducibility

# load dataset
iris = MLDatasets.Iris()
println("First 5 rows of the dataset:")
println(iris.features[1:5, :])  # Check dataset

X = Matrix(iris.features)  # Convert DataFrame to numeric Matrix
println("Size of X: ", size(X))  # Check dimensions
println("First 5 rows of X:")
println(X[1:5, :])



k = 3  # num of clusters

# apply K-means
result = kmeans(X_norm', k; maxiter=100, display=:none)
labels = result.assignments  # cluster labels

println("Cluster assignments:")
println(labels)

# plot using normalized features
scatter(X[:,1], X[:,2], group=labels, markersize=5,
        title="K-means clustering on Iris (Raw Data)",
        xlabel="Sepal Length", ylabel="Sepal Width",
        legend=:topright)

display(plot())


display(plot())  # Show the plot explicitly
readline()  # Keep plot open