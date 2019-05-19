# BFGSMin-Julia
A simple BFGS Minimizer for Julia

This little script is a BFGS miminizer I've constructed mostly to perform MLE in Julia, based in the same code for Python (https://github.com/ighdez/BFGSMin). The code is highly based in the optim(method="BFGS") code from R and its successor Rvmmin() created by John C. Nash.

I attach both the code and an example script to run a Normal linear model regression by MLE, but I also tried with more complicated models such as the Discrete-Continuous Choice Model (DCC) for water demand that I used in my Master's Thesis and performs quite well.

I advice you that the code is very simple, and I'm affraid that quite sloppy. But as you know, is work in progress.

Enjoy and feel free to commit it. :)
