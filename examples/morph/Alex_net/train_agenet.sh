export PYTHONPATH=examples:$PYTHONPATH
build/tools/caffe train --solver examples/morph/Alex_net/solver.pt --gpu 1 --weights examples/morph/bvlc_alexnet.caffemodel 2>&1 | tee examples/morph/Alex_net/age_net.caffelog
