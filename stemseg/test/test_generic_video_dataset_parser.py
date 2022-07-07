import  pytest as pytest

from stemseg.data.generic_video_dataset_parser import GenericVideoSequence


def test_category_rename():
    seq  = GenericVideoSequence(base_dir="some/dir",
                                image_paths=["1.png", "2.png"],
                                height=128,
                                width=256,
                                id="some_id",
                                categories={1: "Person", 2: "Car", 3: "Car"})

    seq.apply_category_id_mapping({"Person":11, "Car":22})
    assert seq.instance_categories == {1: 11, 2: 22, 3: 22}

def test_extract_subsequence():
    seq = GenericVideoSequence(base_dir="some/dir",
                               image_paths=["1.png", "2.png"],
                               height=128,
                               width=256,
                               id="some_id",
                               segmentations=[{2:"some_seq"}, {2:"other_Seq", 1:"asdf"}],
                               categories={1: "Person", 2: "Car", 3: "Car"})

    sub_seq = seq.extract_subsequence([0], new_id="new_id")
    assert sub_seq.instance_categories == {2: "Car"}
    assert len(sub_seq) == 1
    assert sub_seq.segmentations == [seq.segmentations[0]]
    assert sub_seq.image_dims == seq.image_dims
    assert sub_seq.id == "new_id"
    assert sub_seq.image_paths == [seq.image_paths[0]]
    assert sub_seq.base_dir == seq.base_dir
