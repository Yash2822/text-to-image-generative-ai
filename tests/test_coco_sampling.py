from src.data.coco_sampling import COCOSampler


class FakeLoader:
    def __init__(self):
        self.images_by_id = {
            1: {"id": 1, "width": 640, "height": 480},
            2: {"id": 2, "width": 200, "height": 640},
            3: {"id": 3, "width": 640, "height": 200},
            4: {"id": 4, "width": 500, "height": 500},
            5: {"id": 5, "width": 640, "height": 480},
        }

        self.annotations = {
            1: [{"id": 1}],
            2: [{"id": 2}, {"id": 3}],
            3: [{"id": 4}],
            4: [
                {"id": 5},
                {"id": 6},
                {"id": 7},
                {"id": 8},
            ],
            5: [],
        }

    def get_annotations(self, image_id):
        return self.annotations[image_id]


def test_simplest_image():
    sampler = COCOSampler(FakeLoader())

    sample = sampler.simplest_image()

    assert sample.image_id == 1


def test_most_complex_image():
    sampler = COCOSampler(FakeLoader())

    sample = sampler.most_complex_image()

    assert sample.image_id == 4


def test_zero_object_image():
    sampler = COCOSampler(FakeLoader())

    sample = sampler.zero_object_image()

    assert sample.image_id == 5


def test_portrait_image():
    sampler = COCOSampler(FakeLoader())

    sample = sampler.portrait_image()

    assert sample.image_id == 2


def test_landscape_image():
    sampler = COCOSampler(FakeLoader())

    sample = sampler.landscape_image()

    assert sample.image_id == 3


def test_representative_samples_contains_five_samples():
    sampler = COCOSampler(FakeLoader())

    samples = sampler.representative_samples()

    assert len(samples) == 5