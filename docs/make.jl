using Documenter
using Methode_of_conjugate_gradients

makedocs(
    sitename = "Methode_of_conjugate_gradients",
    format = Documenter.HTML(),
    modules = [Methode_of_conjugate_gradients]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.

println(ENV["DOCUMENTER_KEY"])

deploydocs(
    repo = "github.com/lovc21/Methode_of_conjugate_gradients.lj.git",
    push_preview = true,
    devbranch = "master",
)