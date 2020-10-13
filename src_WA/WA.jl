using DataFrames
# using Plots
using PyCall
import PyPlot

PyCall.pygui(:tk)

PyPlot.plot([12,3,4,5,6,7], [2,4,6,7,9,0])
PyPlot.savefig("plots.png")
PyPlot.show()

println("hello world")

readline()