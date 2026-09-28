import unittest

from scripts.data.build_release import clean_row


class ReleaseProjectionTests(unittest.TestCase):
    def test_keeps_training_content_without_original_benchmark_fields(self):
        row = {
            "reader_id": "item-1",
            "title": "Source title",
            "context": "Original problem context",
            "original_answer": "Source answer",
            "reader_target": "A canonical target",
            "variants": [{
                "question": "Generated question",
                "answer": "Generated answer",
                "rewrite_variant_id": "v1",
                "original_question": "Original benchmark question",
                "context": "Original problem context",
            }],
        }
        clean = clean_row(row)
        self.assertEqual(clean["reader_target"], row["reader_target"])
        self.assertEqual(clean["variants"], [{
            "question": "Generated question",
            "answer": "Generated answer",
            "rewrite_variant_id": "v1",
        }])
        self.assertNotIn("title", clean)
        self.assertNotIn("context", clean)
        self.assertNotIn("original_answer", clean)

    def test_rejects_internal_paths(self):
        with self.assertRaisesRegex(ValueError, "private machine path"):
            clean_row({"reader_id": "item-1", "notes":
                       "/mnt/shared-storage-user/account/run"})

    def test_rejects_incomplete_variant(self):
        with self.assertRaisesRegex(ValueError, "question or answer"):
            clean_row({"reader_id": "item-1", "variants": [{
                "question": "Question only"
            }]})


if __name__ == "__main__":
    unittest.main()
