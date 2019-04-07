import collections
import six
import tensorflow as tf
import numpy as np
import os

FLAGS = tf.app.flags.FLAGS

'''
cifar-10 directory structure:
data_dir
  -test
  -train
  -test_lst.txt
  -train_lst.txt
'''

tf.app.flags.DEFINE_string('data_dir', '/media/data/cifar10', 'Folder containing data.')
tf.app.flags.DEFINE_string('test_lst', '/media/data/cifar10/test_lst.txt', 'Testing images list')
tf.app.flags.DEFINE_string('train_lst', '/media/data/cifar10/train_lst.txt', 'Training images list')
tf.app.flags.DEFINE_string('output_dir', './tfrecords', 'Path to save tfrecords.')

_HEIGHT = 32
_WIDTH = 32
_CHANNEL = 3
_NUM_SHARDS_TEST = 10
_NUM_SHARDS_TRAIN = 50

def _int64_list_feature(values):
    """Returns a TF-Feature of int64_list.
    Args:
    values: A scalar or list of values.
    Returns:
    A TF-Feature.
    """
    if not isinstance(values, collections.Iterable):
        values = [values]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _float_list_feature(values):

    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _bytes_list_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
    values: A string.
    Returns:
    A TF-Feature.
    """
    def norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(image_data, filename, height, width, channel, label):
    """Converts one image/segmentation pair to tf example.
    Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.
    Returns:
    tf example of one image/segmentation pair.
    """
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_list_feature(image_data),
        'image/filename': _bytes_list_feature(filename),
        'image/height': _int64_list_feature(height),
        'image/width': _int64_list_feature(width),
        'image/channels': _int64_list_feature(channel),
        'image/label': _int64_list_feature(label)
    }))


def _convert_dataset(lst, num_shards, out_name, shuffle=False):


    with open(lst) as f:
        lines = f.readlines()
    num_images = len(lines)
    num_per_shard = int(np.ceil(num_images / float(num_shards)))

    if shuffle:
        np.random.shuffle(lines)

    for shard_id in range(num_shards):
        print('processing shard %05d' % shard_id)
        output_filename = os.path.join(FLAGS.output_dir, 
            '%s-%05d-of-%05d.tfrecord' % (out_name, shard_id, num_shards))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                img_path_, label_ = lines[i].strip().split(' ')
                filename_ = img_path_.split('/')[-1]
                img_path_ = os.path.join(FLAGS.data_dir, img_path_)
                label_ = int(label_)
                image_data = tf.gfile.FastGFile(img_path_, 'rb').read()
                example = image_seg_to_tfexample(image_data, filename_, 
                    _HEIGHT, _WIDTH, _CHANNEL, label_)
                tfrecord_writer.write(example.SerializeToString())


def main():

    # data for testing
    print('making testing dataset...')
    _convert_dataset(FLAGS.test_lst, _NUM_SHARDS_TEST, 'test/cifar10-test')
    
    # data for training
    print('making training dataset...')
    _convert_dataset(FLAGS.train_lst, _NUM_SHARDS_TRAIN, 'train/cifar10-train', True)


if __name__ == '__main__':
    main()