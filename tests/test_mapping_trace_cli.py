from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from aloi.cli.compile import main
from tests.helpers import compilation


class MappingTraceAndCLITest(unittest.TestCase):
    def test_schedule_override_reaches_aim(self) -> None:
        result = compilation()
        wq = next(
            op
            for op in result.aim.get_function().operations
            if op.attrs.get("weight") == "wq" and op.op == "aim.pim.mac_dram_gb"
        )
        self.assertEqual(wq.attrs["mapping"]["tile"]["in_feature"], 256)
        self.assertEqual(wq.attrs["mapping"]["bind"]["out_feature"], "channel")

    def test_trace_contains_real_isr_and_semantic_comments(self) -> None:
        trace = compilation().trace
        self.assertIn("AiM WR_GB", trace)
        self.assertIn("AiM MAC_ABK", trace)
        self.assertIn("AiM RD_MAC", trace)
        self.assertIn("# ALOI_PNM", trace)
        self.assertIn("# ALOI_CXL", trace)
        self.assertTrue(trace.endswith("AiM EOC\n"))

    def test_reference_cli_emits_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            status = main(
                [
                    "examples/llama2_70b",
                    "--seq-len",
                    "32",
                    "--tp",
                    "8",
                    "--device",
                    "configs/devices/cent.yaml",
                    "--schedule",
                    "examples/llama2_70b/schedule.py",
                    "--dump-ir",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            expected = {
                "exported_program.txt",
                "01_semantic.aloi",
                "02_sharded.aloi",
                "03_device.aloi",
                "04_aim.aloi",
                "05_aim.trace",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertIn("aloi.linear", (output / "01_semantic.aloi").read_text())
            self.assertNotIn(" = aloi.linear ", (output / "03_device.aloi").read_text())


if __name__ == "__main__":
    unittest.main()
