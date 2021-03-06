import os
import argparse
import cv2
import argparse
import re
import numpy as np
import sys
import time

sys.path.append('../evolving_wilds')
from cnn_utils import image_utils, vis_utils

from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('inDir', nargs='*', type=str, help='Dir to get images from')
    ap.add_argument('-prefixes', nargs='?', default=['train', 'test', 'eval'], type=str,
                    help='Allowed prefixes i.e. train test')
    ap.add_argument('-max_epoch', nargs='?', default=0, type=int, help='Max epoch number to include in plots')
    ap.add_argument('-range', nargs=2, default=[None, None], type=int, help='Start and end of epoch range to show')
    ap.add_argument('-out_names', nargs='*', default=[], type=str, help='Name of out file without extension')
    ap.add_argument('-out_dir', default='../public_html/lpat', type=str)
    ap.add_argument('-span', nargs='?', default=500, help='Number of frames to look back')
    args = ap.parse_args()
    prefixes = args.prefixes

    if len(args.out_names) < len(args.inDir):
        args.out_names += [None] * (len(args.inDir) - len(args.out_names))

    for dirCount, inDir in enumerate(args.inDir):
        if args.out_names is not None and args.out_names[dirCount] is not None:
            out_name = args.out_names[dirCount]
        else:
            print(inDir)
            out_name = os.path.basename(os.path.normpath(inDir))
            print(out_name)
        if len(os.listdir(inDir)) == 0:
            continue

        # look for train, test, eval ims
        out_ims_all = []
        print(out_name)
        for prefix in prefixes:
            starttime = time.time()
            n_epochs_to_show = 3
            n_batches_to_show = 5

            imFiles = [f for f in os.listdir(inDir) if
                       (f.endswith('.png') or f.endswith('.jpg')) and 'epoch' and prefix in f]

            if len(imFiles) == 0:
                print('Found no images in {}! Skipping...'.format(inDir))
                continue

            epochNums = np.unique([int(re.search('(?<=epoch)[0-9]*(?=_)', f).group(0)) for f in imFiles])

            if args.range[0] is not None:
                maxEpochNum = args.range[1]
                startEpoch = args.range[0]
            elif args.max_epoch == 0:
                maxEpochNum = max(epochNums)
                startEpoch = max(min(maxEpochNum, maxEpochNum - args.span), min(epochNums))
            else:
                maxEpochNum = args.max_epoch
                startEpoch = max(min(maxEpochNum, maxEpochNum - args.span), min(epochNums))
            print('Parsing {} ims from epoch {} to {}'.format(prefix, startEpoch, maxEpochNum))

            epochsInRange = [e for e in epochNums if e >= startEpoch and e <= maxEpochNum]
            # only show as many epochs as are available in the folder
            n_epochs_to_show = min(len(epochsInRange), n_epochs_to_show)
            curr_prefix_im_files = np.empty((n_epochs_to_show, n_batches_to_show), dtype=object)

            epoch_count = 0

            epochIdxs = np.linspace(0, len(epochsInRange) - 1, n_epochs_to_show, dtype=int)

            epochsToShow = []
            oddEpochsInRange = [e for e in epochsInRange if e % 2 == 1]
            for i, idx in enumerate(epochIdxs):
                epochToShow = epochsInRange[i]
                if i % 2 == 1 and epochToShow % 2 == 0 and len(oddEpochsInRange) > 0:
                    epochToShow = oddEpochsInRange[np.argmin(np.abs(oddEpochsInRange - epochToShow))]
                epochsToShow.append(epochToShow)

            epochsToShow = [epochsInRange[i] for i in epochIdxs]
            print(epochsToShow)
            for en in epochsToShow:
                curr_epoch_ims = [os.path.join(inDir, f) for f in imFiles if 'epoch' + str(en) + '_' in f]

                batch_count = 0
                curr_n_batches_to_show = min(len(curr_epoch_ims), n_batches_to_show)
                for bi in np.linspace(0, len(curr_epoch_ims) - 1, curr_n_batches_to_show, dtype=int):
                    curr_prefix_im_files[epoch_count, batch_count] = curr_epoch_ims[bi]
                    print(bi)
                    if bi == 0:
                        im = cv2.imread(curr_epoch_ims[bi])
                        R = im.shape[0] * max(1, int(480.0 / im.shape[1]))
                        C = im.shape[1] * max(1, int(480 / im.shape[1]))

                    batch_count += 1
                epoch_count += 1
            curr_prefix_im_files = curr_prefix_im_files.transpose()
            print('Parsing {} ims took {}s'.format(prefix, time.time() - starttime))
            starttime = time.time()

            # remove rows that are completely empty
            rows_to_delete = []
            for r in range(curr_prefix_im_files.shape[0]):
                if np.all([curr_prefix_im_files[r, c] is None for c in range(curr_prefix_im_files.shape[1])]):
                    rows_to_delete.append(r)
            curr_prefix_im_files = np.delete(curr_prefix_im_files, rows_to_delete, axis=0)
            # assume all ims are of the same aspect ratio. We want each im to be at least 480px tall
            outIm = np.zeros((curr_prefix_im_files.shape[0] * R, curr_prefix_im_files.shape[1] * C, 3))
            # im files is nest list of prefix, epoch nums, batches
            row_count = 0
            for im_row in curr_prefix_im_files:
                im_count = 0

                for im_file in im_row:
                    if im_file is None or not os.path.isfile(im_file):
                        im_count += 1
                        continue
                    im = cv2.imread(im_file)
                    curr_im = cv2.resize(im, None, fx=C / float(im.shape[1]), fy=R / float(im.shape[0]))

                    im_pil = Image.fromarray(curr_im.astype(np.uint8))
                    draw = ImageDraw.Draw(im_pil)

                    text = os.path.basename(im_file)
                    font_size = min(30, int(3. * C / len(text)))
                    font = ImageFont.truetype('Ubuntu-M.ttf', font_size)

                    draw.text((10, 2), '{}'.format(text), font=font, fill=(50, 120, 255))

                    curr_im = np.asarray(im_pil)

                    outIm[int(row_count * R):int((row_count + 1) * R), int(im_count * C):int((im_count + 1) * C),
                    :] = curr_im
                    im_count += 1
                row_count += 1

            for c in range(n_epochs_to_show):
                cv2.line(outIm, ((c + 1) * C, 0), ((c + 1) * C, outIm.shape[0]), color=(0, 0, 255), thickness=5)
            out_ims_all.append(outIm)
            print('making {} ims took {}s'.format(prefix, time.time() - starttime))

        if len(out_ims_all) == 0:
            continue

        starttime = time.time()
        maxR = 0
        maxC = 0
        for i in range(len(out_ims_all)):
            if out_ims_all[i].shape[0] > maxR:
                maxR = out_ims_all[i].shape[0]
            if out_ims_all[i].shape[1] > maxC:
                maxC = out_ims_all[i].shape[1]

        out_im = vis_utils.concatenate_with_pad(out_ims_all, axis=0)

        # draw a black line separating rows of images
        im_row_delimiters = [im.shape[0] for im in out_ims_all[:-1]]
        for d in im_row_delimiters:
            out_im[d:d + 5, :, :] = 0
        print('Combining ims took {}s'.format(time.time() - starttime))

        outFile = os.path.join(args.out_dir, out_name + '.jpg')
        print('Writing {} rows to {} im from dir {} to out im {}\n'.format(row_count, out_im.shape, inDir, outFile))

        cv2.imwrite(outFile, out_im)
