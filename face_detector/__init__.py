
from face_detector.haarcascades_detector import HaarcascadeDetector


def build_detector(args):
    if args.face_detect_algorithm == 'haarcascade':
        detector = HaarcascadeDetector(args.pretrained_haarcascade_path)
    else:
        raise NotImplementedError(f'Face detector for {args.face_detect_algorithm} is not implemented yet.')
    return detector
