from absl.testing import absltest
from jax._src import test_util as jtu
from jax import jit
from jax import config
import numpy as np
from jax._src.cudnn import cudnn_fusion

config.parse_flags_with_absl()

class CudnnFusionTest(jtu.JaxTestCase):
    def test_cudnn_fusion(self):

        @cudnn_fusion
        def fusion_computation1(a, b, c):
            return a - b * c

        jit_kernel = jit(fusion_computation1)
        lowered = jit_kernel.lower(1, 2, 3)
        hlo = lowered.as_text("hlo")
        stablehlo = lowered.as_text("stablehlo")

        print(stablehlo)
        self.assertIn("func.func private @fusion_computation1", stablehlo)

        print(hlo)
        self.assertIn("custom-call", hlo)
        self.assertIn("called_computations=", hlo)

        # TODO: lower to a normal call on CPU
        # assert jit_kernel(1, 2, 3) == 1 - 2 * 3

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
