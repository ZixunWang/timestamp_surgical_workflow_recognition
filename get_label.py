import os


def main():
    anno_dir = './dataset/cholec80/phase_annotations/'
    anno_files = [os.path.join(anno_dir, x) for x in sorted(os.listdir(anno_dir))]
    target_files = [os.path.join('./dataset/cholec80/frames_annotations', str(x.split('.')[0]+'.txt')) for x in sorted(os.listdir(anno_dir))]
    for anno_file, target_file in zip(anno_files, target_files):
        with open(anno_file, 'r') as f:
            content = f.read().split('\n')[1:-1]
        labels = [x.split('\t')[1] for x in content][::25]
        with open(target_file, 'w') as f:
            for idx, label in enumerate(labels):
                f.write('{}\t{}\n'.format(idx, label))


if __name__ == '__main__':
    main()
