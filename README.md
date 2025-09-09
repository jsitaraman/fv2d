

This is a 2-D finite volume unstructured grid code
to serve as a educational demonstrator for creating graph
neural network based linear solver augmentation.

Currently it initializes an isentropic vortex and 
solves inviscid Euler equations in 2D. But can be
extended.

 -  2nd order LSQ gradients
 -  Roe/Lax flux schemes
 -  Roe Jacoabians 
 -  Implicit BDF2 and RK3 explicit

What is interesting about this code?

This was a vibe-coding excercise.

Recently and intern of mine said I was a traditionalist
in the "trough of disillusionment" and I wanted to 
get out of that trough and see how the new world is.

Total time spend developing the code was close
to 8 hours. It was done over evenings and weekends 
as an excercise to learn vibe-coding.
Bulk of the code was written by chatgpt
with modifications to fix some of the mistakes.

It is *in theory* performance portable, i.e. would
work with numpy and cupy. This is yet to be tested.


Jay Sitaraman
09/09/2025
