# quantiles-biased-python-experimental

This package implements a quantile approximation sketch similar to that introduced in "Optimal Quantile Approximation in Streams" by Zohar Karnin, Kevin Lang, Edo Liberty.
http://arxiv.org/abs/1603.05346

Given a stream of comparable items, given in arbitrary order, the sketch provides the quantiles of the items. The implementation is in python, and allows for various modifications of the original
KLL algorithm. The most notable knob is the ability to use biased estimation, achieving much better performance on large quantiles such as p99, p99.5, etc., at the expense of slightly worse guarantees 
for the smaller quantiles.

This package is experimental and will be subject to changes.


