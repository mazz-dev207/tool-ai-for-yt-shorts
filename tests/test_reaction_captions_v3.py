import tempfile
import unittest
from pathlib import Path

from src.caption.grouping import Word, WordGroup
from src.reaction_captions.detector import (
    _detect_transcript_events,
    _merge_same_kind,
)
from src.reaction_captions.models import ReactionEvent
from src.reaction_captions.renderer import append_reaction_captions_to_ass


class ReactionCaptionV3Tests(unittest.TestCase):
    def group(self, *words):
        items = [Word(word=text, start=start, end=end) for text, start, end in words]
        return WordGroup(words=items, start=items[0].start, end=items[-1].end)

    def ass_file(self, root: Path) -> Path:
        path = root / "clip.ass"
        path.write_text(
            "[Script Info]\nPlayResX: 1080\nPlayResY: 1920\n\n"
            "[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,"
            "OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,"
            "ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,"
            "MarginR,MarginV,Encoding\n"
            "Style: Default,Anton,92,&H00FFFFFF,&H0000FFFF,&H00000000,"
            "&H00000000,-1,0,0,0,100,100,0,0,1,4,1,5,120,120,0,1\n\n"
            "[Events]\n"
            "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,HELLO\n",
            encoding="utf-8",
        )
        return path

    def test_repeated_laughter_token_becomes_caption(self):
        groups = [self.group(("HAHAHA", 1.0, 1.5))]
        events = _detect_transcript_events(groups)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "laugh")
        self.assertEqual(events[0].text, "[LAUGHS]")

    def test_bracketed_gasp_annotation_becomes_caption(self):
        groups = [self.group(("[GASPS]", 2.0, 2.4))]
        events = _detect_transcript_events(groups)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].kind, "gasp")
        self.assertEqual(events[0].text, "[GASPS]")

    def test_plain_word_laughing_is_not_assumed_to_be_real_laughter(self):
        groups = [self.group(("laughing", 2.0, 2.4))]
        events = _detect_transcript_events(groups)
        self.assertEqual(events, [])

    def test_adjacent_same_reactions_merge(self):
        events = _merge_same_kind(
            [
                ReactionEvent("laugh", "[LAUGHS]", 1.0, 1.8, 0.70),
                ReactionEvent("laugh", "[LAUGHS]", 2.0, 2.7, 0.82),
            ]
        )
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].start, 1.0)
        self.assertAlmostEqual(events[0].end, 2.7)
        self.assertAlmostEqual(events[0].score, 0.82)

    def test_renderer_adds_separate_layer_three_style(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.ass_file(Path(tmp))
            events = [ReactionEvent("laugh", "[LAUGHS]", 1.0, 2.0, 0.91)]
            self.assertTrue(append_reaction_captions_to_ass(path, events))
            content = path.read_text(encoding="utf-8")
            self.assertIn("Style: ReactionCaption,", content)
            self.assertIn("Dialogue: 3,", content)
            self.assertIn("[LAUGHS]", content)
            self.assertIn("Dialogue: 0,", content)

    def test_renderer_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.ass_file(Path(tmp))
            events = [ReactionEvent("gasp", "[GASPS]", 1.0, 2.0, 0.91)]
            append_reaction_captions_to_ass(path, events)
            append_reaction_captions_to_ass(path, events)
            content = path.read_text(encoding="utf-8")
            self.assertEqual(content.count("Style: ReactionCaption,"), 1)
            self.assertEqual(content.count(",ReactionCaption,"), 1)

    def test_empty_detection_removes_only_reaction_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.ass_file(Path(tmp))
            events = [ReactionEvent("gasp", "[GASPS]", 1.0, 2.0, 0.91)]
            append_reaction_captions_to_ass(path, events)
            append_reaction_captions_to_ass(path, [])
            content = path.read_text(encoding="utf-8")
            self.assertNotIn("ReactionCaption", content)
            self.assertIn("Dialogue: 0,", content)


if __name__ == "__main__":
    unittest.main()
