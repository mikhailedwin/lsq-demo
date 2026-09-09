"""AvatarSpec is the contract between the API's avatar catalog and the face backends."""

from __future__ import annotations

from qav_face.avatars import AvatarSpec, AvatarStyle


def test_catalog_entry_maps_to_a_spec() -> None:
    spec = AvatarSpec.from_catalog(
        {
            "id": "mt-yongen",
            "name": "Yongen",
            "renderer": "musetalk",
            "assets": {"avatarDir": "yongen"},
        },
        width=512,
        height=512,
        fps=25,
    )
    assert (spec.id, spec.renderer, spec.assets["avatarDir"]) == ("mt-yongen", "musetalk", "yongen")


def test_missing_renderer_falls_back_to_the_deployment_default() -> None:
    spec = AvatarSpec.from_catalog({"id": "x"}, width=256, height=256, fps=25, default_renderer="musetalk")
    assert spec.renderer == "musetalk"
    spec = AvatarSpec.from_catalog(None, avatar_id="y", width=256, height=256, fps=25)
    assert spec.renderer == "procedural" and spec.id == "y"


def test_style_accepts_both_camel_and_snake_case() -> None:
    assert AvatarStyle.from_dict({"hairStyle": "bun"}).hair_style == "bun"
    assert AvatarStyle.from_dict({"hair_style": "bald"}).hair_style == "bald"
    assert AvatarStyle.from_dict(None).hair_style == "short"


def test_output_size_and_fps_come_from_the_session_not_the_catalog() -> None:
    entry = {"id": "a", "renderer": "musetalk", "assets": {"avatarDir": "yongen"}}
    small = AvatarSpec.from_catalog(entry, width=256, height=256, fps=25)
    big = AvatarSpec.from_catalog(entry, width=704, height=384, fps=30)
    assert (small.width, small.fps) == (256, 25)
    assert (big.width, big.height, big.fps) == (704, 384, 30)
