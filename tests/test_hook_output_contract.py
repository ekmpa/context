import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts import hook


def _has_hook_backend() -> bool:
    try:
        import duckdb  # noqa: F401

        return True
    except Exception:
        pass

    try:
        import pyarrow.dataset  # noqa: F401

        return True
    except Exception:
        return False


@unittest.skipUnless(_has_hook_backend(), "Requires duckdb or pyarrow backend")
class TestHookOutputContract(unittest.TestCase):
    def _write_edges_csv(self, rows):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        path = Path(tmp.name)
        try:
            tmp.write("src,dst\n")
            for src, dst in rows:
                tmp.write(f"{src},{dst}\n")
            tmp.flush()
        finally:
            tmp.close()
        return path

    def test_temporal_mode_single_domain_output_has_all_months_and_lists(self):
        edges = self._write_edges_csv(
            [
                ("com.example", "org.factcheck"),
                ("net.news", "com.example"),
                ("com.other", "com.else"),
            ]
        )

        cmd = [
            "python",
            "scripts/hook.py",
            "example.com",
            "--mode",
            "temporal",
            "--month-file",
            f"oct2024={edges}",
        ]
        completed = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)
        self.assertEqual(set(payload.keys()), set(hook.MONTHS))
        for month in hook.MONTHS:
            self.assertIsInstance(payload[month], list)
            self.assertTrue(all(isinstance(x, str) for x in payload[month]))

        self.assertEqual(payload["oct2024"], ["net.news", "org.factcheck"])
        self.assertEqual(payload["nov2024"], [])

    def test_latest_mode_default_returns_two_hop_neighbors_from_most_recent(self):
        edges = self._write_edges_csv(
            [
                ("com.seed", "org.a"),
                ("net.b", "com.seed"),
                ("org.a", "io.c"),
            ]
        )

        cmd = [
            "python",
            "scripts/hook.py",
            "seed.com",
            "--month-file",
            f"oct2024={edges}",
        ]
        completed = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )

        payload = json.loads(completed.stdout)
        self.assertIsInstance(payload, list)
        self.assertTrue(all(isinstance(x, str) for x in payload))
        self.assertEqual(payload, ["io.c", "net.b", "org.a"])


if __name__ == "__main__":
    unittest.main()
