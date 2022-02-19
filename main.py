
import argparse
import logging

import torch
import cv2

from face_detector import build_detector

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()

    # source input, output
    parser.add_argument(
        '--use_webcam', action='store_true',
        help='Use webcam as video source. Prioritize using webcam when video path is given together'
    )
    parser.add_argument('--video_source_path', type=str, required=False, help='Video source path')
    parser.add_argument('--output_path', type=str, required=False, help='Video save path')

    # reading video
    parser.add_argument('--horizontal_rescale_dim', type=int, default=858)
    parser.add_argument('--vertical_rescale_dim', type=int, default=480)
    parser.add_argument('--frame_rate', type=float, default=24.0)

    # detect algorithm
    parser.add_argument(
        '--face_detect_algorithm', type=str, default='haarcascade', choices=['haarcascade']
    )
    parser.add_argument(
        '--pretrained_haarcascade_path', type=str,
        default='face_detector/pretrained_data/haarcascade_frontalcatface.xml'
    )

    # device
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.device != 'cpu' and not torch.cuda.is_available():
        logging.warning('CUDA is not available. Set device as cpu.')
        args.device = 'cpu'
    return args


def main():
    args = parse_args()

    if args.use_webcam:
        video_source = 0
        logging.info("Using webcam")
    elif args.video_source_path:
        video_source = args.video_source_path
        logging.info("Using video source")
    else:
        raise IOError("Video source does not exist. Check source path.")

    detector = build_detector(args)
    cap = cv2.VideoCapture(video_source)
    frame_size = (args.horizontal_rescale_dim, args.vertical_rescale_dim)
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    if args.output_path:
        writer = cv2.VideoWriter(args.output_path, fourcc=fourcc, fps=args.frame_rate, frameSize=frame_size)

    if not cap.isOpened():
        raise IOError("Cannot read video source")
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        frame = cv2.resize(src=frame, dsize=frame_size, interpolation=cv2.INTER_AREA)
        # face detection
        faces = detector.detect_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if args.output_path:
            writer.write(frame)
        cv2.imshow('Video', frame)
        c = cv2.waitKey(1)
        if c == 27:  # == esc key
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
