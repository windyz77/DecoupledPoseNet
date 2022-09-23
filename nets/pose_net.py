# coding=utf-8
from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
"""
Adopted from https://github.com/tinghuiz/SfMLearner
"""


def pose_exp_net(tgt_image, src_image):
    inputs = tf.concat([tgt_image, src_image], axis=3)
    with tf.variable_scope('pose_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
                normalizer_fn=None,
                weights_regularizer=slim.l2_regularizer(0.0004),
                activation_fn=tf.nn.relu,
                outputs_collections=end_points_collection):
            cnv1 = slim.conv2d(inputs, 16, [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1, 16, [7, 7], stride=1, scope='cnv1b')
            cnv2 = slim.conv2d(cnv1b, 32, [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2, 32, [5, 5], stride=1, scope='cnv2b')
            cnv3 = slim.conv2d(cnv2b, 64, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3, 64, [3, 3], stride=1, scope='cnv3b')
            cnv4 = slim.conv2d(cnv3b, 128, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4, 128, [3, 3], stride=1, scope='cnv4b')
            cnv5 = slim.conv2d(cnv4b, 256, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5, 256, [3, 3], stride=1, scope='cnv5b')

            # Pose specific layers
            cnv6 = slim.conv2d(cnv5b, 256, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6, 256, [3, 3], stride=1, scope='cnv6b')
            cnv7 = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7, 256, [3, 3], stride=1, scope='cnv7b')
            pose_pred = slim.conv2d(
                cnv7b,
                6, [1, 1],
                scope='pred',
                stride=1,
                normalizer_fn=None,
                activation_fn=None)
            pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            # Empirically we found that scaling by a small constant on the 
            # translations facilitates training.
            pose_final = tf.reshape(pose_avg, [-1, 6])
            pose_final = tf.concat(
                [pose_final[:, 0:3], 0.01 * pose_final[:, 3:6]], axis=1)
    return pose_final

def pose_net_loop_trans(posenet_inputs, is_training, variable_reuse=None):
    bs = posenet_inputs.get_shape().as_list()[0]
    # is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    seq_length = 1
    with tf.variable_scope('pose_net_trans', reuse=variable_reuse) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2 = slim.conv2d(conv1, 32,  5, 2)
            conv3 = slim.conv2d(conv2, 64,  3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred_t = slim.conv2d(conv7, 3 * seq_length, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg_t = tf.reduce_mean(pose_pred_t, [1, 2])
            # 获得 [tx ty tz]
            # pose_final_t = 0.01 * tf.reshape(pose_avg_t, [-1, seq_length, 3])
            pose_final_t = tf.reshape(pose_avg_t, [-1, seq_length, 3])

            inital_roate = tf.constant([[[0., 0., 0.]]])
            pose_final_r = tf.tile(inital_roate, [bs, seq_length, 1])
            # [tx ty tz] --> [tx ty tz 0 0 0 1]
            pose_final = tf.concat([pose_final_t, pose_final_r], axis=2)
            pose_final = tf.squeeze(pose_final, axis=[1])
            return pose_final

def pose_net_loop_trans_refine(posenet_inputs, is_training, variable_reuse=None):
    bs = posenet_inputs.get_shape().as_list()[0]
    # is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    seq_length = 1
    with tf.variable_scope('pose_net_trans_refine', reuse=variable_reuse) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2 = slim.conv2d(conv1, 32,  5, 2)
            conv3 = slim.conv2d(conv2, 64,  3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred_t = slim.conv2d(conv7, 3 * seq_length, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg_t = tf.reduce_mean(pose_pred_t, [1, 2])
            # [tx ty tz]
            # pose_final_t = 0.01 * tf.reshape(pose_avg_t, [-1, seq_length, 3])
            pose_final_t = tf.reshape(pose_avg_t, [-1, seq_length, 3])

            inital_roate = tf.constant([[[0., 0., 0.]]])
            pose_final_r = tf.tile(inital_roate, [bs, seq_length, 1])
            # [tx ty tz 0 0 0 1]
            pose_final = tf.concat([pose_final_t, pose_final_r], axis=2)
            pose_final = tf.squeeze(pose_final, axis=[1])
            return pose_final

def pose_net_loop_roate_quater(posenet_inputs, is_training, variable_reuse=None):
    bs = posenet_inputs.get_shape().as_list()[0]
    # is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    seq_length = 1
    with tf.variable_scope('pose_net_roate', reuse=variable_reuse) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2 = slim.conv2d(conv1, 32,  5, 2)
            conv3 = slim.conv2d(conv2, 64,  3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred_r = slim.conv2d(conv7, 4 * seq_length, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg_r = tf.reduce_mean(pose_pred_r, [1, 2])
            # [x y z w]
            pose_final_r = 0.01 * tf.reshape(pose_avg_r, [-1, seq_length, 4])

            inital_roate = tf.constant([[[0., 0., 0., 1.]]])
            inital_roate = tf.tile(inital_roate, [bs, seq_length, 1])
            # # [x y z w+1]
            pose_final_r = inital_roate + pose_final_r

            quater = pose_final_r
            bs, num, _ = quater.get_shape().as_list()
            norm_quater = tf.reshape(tf.norm(quater, axis=2), shape=[bs, num, 1])
            pose_final_r = tf.divide(quater, norm_quater + 1e-10)
            # [0 0 0]
            pose_final_t = tf.zeros([bs, seq_length, 3])
            # [0 0 0 x y z w+1]
            pose_final = tf.concat([pose_final_t, pose_final_r], axis=2)
            pose_final = tf.squeeze(pose_final, axis=[1])
            return pose_final

def pose_net_loop_roate_refine_quater(posenet_inputs, is_training, variable_reuse=None):
    bs = posenet_inputs.get_shape().as_list()[0]
    # is_training = opt.mode == 'train_rigid'
    batch_norm_params = {'is_training': is_training}
    seq_length = 1
    with tf.variable_scope('pose_net_roate_refine', reuse=variable_reuse) as sc:
        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0001),
                            activation_fn=tf.nn.relu):
            conv1 = slim.conv2d(posenet_inputs, 16,  7, 2)
            conv2 = slim.conv2d(conv1, 32,  5, 2)
            conv3 = slim.conv2d(conv2, 64,  3, 2)
            conv4 = slim.conv2d(conv3, 128, 3, 2)
            conv5 = slim.conv2d(conv4, 256, 3, 2)
            conv6 = slim.conv2d(conv5, 256, 3, 2)
            conv7 = slim.conv2d(conv6, 256, 3, 2)
            pose_pred_r = slim.conv2d(conv7, 4 * seq_length, 1, 1,
                                    normalizer_fn=None, activation_fn=None)
            pose_avg_r = tf.reduce_mean(pose_pred_r, [1, 2])
            pose_final_r = 0.01 * tf.reshape(pose_avg_r, [-1, seq_length, 4])
            inital_roate = tf.constant([[[0., 0., 0., 1.]]])
            inital_roate = tf.tile(inital_roate, [bs, seq_length, 1])

            pose_final_r = inital_roate + pose_final_r

            quater = pose_final_r
            bs, num, _ = quater.get_shape().as_list()
            norm_quater = tf.reshape(tf.norm(quater, axis=2), shape=[bs, num, 1])
            pose_final_r = tf.divide(quater, norm_quater + 1e-10)

            pose_final_t = tf.zeros([bs, seq_length, 3])
            pose_final = tf.concat([pose_final_t, pose_final_r], axis=2)
            pose_final = tf.squeeze(pose_final, axis=[1])
            return pose_final

from utils.utils import *
def build_pose_list_quater(refine_pose):
    if not refine_pose['is_matrix']:
        refine_pose['pose'] = pose_vec2mat_quater(refine_pose['pose'][:, :])
        refine_pose['is_matrix'] = True
    return refine_pose

def pose_vec2mat_quater(vec):
  batch_size = vec.get_shape().as_list()[0]

  translation = tf.slice(vec, [0, 0], [-1, 3])  # [B, 3]
  translation = tf.expand_dims(translation, -1)  # [B, 3, 1]

  rx = tf.slice (vec, [0, 3], [-1, 1])  # [B, 1]
  ry = tf.slice (vec, [0, 4], [-1, 1])  # [B, 1]
  rz = tf.slice (vec, [0, 5], [-1, 1])  # [B, 1]
  rw = tf.slice (vec, [0, 6], [-1, -1])  # [B, 1]
  rot_mat = quater2mat(rx, ry, rz, rw)
  # print(rot_mat)
  # trans_mat = tf.reshape(tf.eye(3), shape=[1, 3, 3])
  # trans_mat = tf.tile(trans_mat, [batch_size, 1, 1])

  filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
  filler = tf.tile(filler, [batch_size, 1, 1])

  transform_mat = tf.concat([rot_mat, translation], axis=2)
  transform_mat = tf.concat([transform_mat, filler], axis=1)

  return transform_mat

def quater2mat(x, y, z, w):
  B, _ = z.get_shape().as_list()
  zeros = tf.zeros([B, 1])
  ones = tf.reshape(tf.eye(3), [1, 3, 3])
  ones = tf.tile(ones, [B, 1, 1])
  # print (-z)
  rotx_1 = tf.concat([zeros, -z, y], axis=1)  # [4, 3]
  rotx_2 = tf.concat([z, zeros, -x], axis=1)  # [4, 3]
  rotx_3 = tf.concat([-y, x, zeros], axis=1)  # [4, 3]
  vx_mat = tf.stack([rotx_1, rotx_2, rotx_3], axis=1)  # [4, 3, 3]

  rotw_1 = tf.concat([zeros, tf.multiply(-z, w), tf.multiply(y, w)], axis=1)  # [4, 3]
  rotw_2 = tf.concat([tf.multiply(z, w), zeros, tf.multiply(-x, w)], axis=1)  # [4, 3]
  rotw_3 = tf.concat([tf.multiply(-y, w), tf.multiply(x, w), zeros], axis=1)  # [4, 3]

  vw_mat = tf.stack([rotw_1, rotw_2, rotw_3], axis=1)  # [4, 3, 3]
  vv_mat = tf.matmul(vx_mat, vx_mat)
  # rotMat = tf.add(tf.multiply(tf.add(vv_mat, vw_mat), 2), ones)
  rotMat = tf.add(tf.add(vv_mat, vw_mat) * 2, ones)

  return rotMat

def build_pose_refine_quater(init_pose, refine_pose, refine_trpe='T'):
    refined_pose = {}
    init_pose = build_pose_list_quater(init_pose)
    refine_pose = build_pose_list_quater(refine_pose)
    if refine_trpe == 'T':
        refined_pose['pose'] = tf.matmul(refine_pose['pose'], init_pose['pose'])
        refined_pose['is_matrix'] = True
    elif refine_trpe == 'R':
        refined_pose['pose'] = tf.matmul(init_pose['pose'], refine_pose['pose'])
        refined_pose['is_matrix'] = True
    return refined_pose

def build_posenet_quater_loop(image1, image2, fwd_warp_parameter, bwd_warp_parameter, is_training=True,
                               variable_reuse=False):

    # *****************************************************************************************************************
    reuse_variables = False if is_training else True
    with tf.variable_scope('pose_net', reuse=reuse_variables):

        '''平移'''
        posenet_inputs = tf.concat([image1, image2], axis=3)
        init_trans_vec = {'pose': pose_net_loop_trans(posenet_inputs, is_training, variable_reuse), 'is_matrix': False}
        pose_mat_step1 = build_pose_list_quater(init_trans_vec)

        fwd_warp_parameter['pose'] = pose_mat_step1['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step1 = transformer_old(image2, depth_flow, [256, 832])
        # *****************************************************************************************************************

        # *****************************************************************************************************************
        '''平移优化'''
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step1], axis=3)
        delta_trans_vec = {'pose': pose_net_loop_trans_refine(posenet_inputs, is_training, variable_reuse), 'is_matrix': False}
        pose_mat_step2 = build_pose_refine_quater(pose_mat_step1, delta_trans_vec, refine_trpe='T')

        fwd_warp_parameter['pose'] = pose_mat_step2['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step2 = transformer_old(image2, depth_flow, [256, 832])
        # *****************************************************************************************************************

        # *****************************************************************************************************************
        '''旋转'''
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step2], axis=3)
        init_roate_vec = {'pose': pose_net_loop_roate_quater(posenet_inputs, is_training, variable_reuse), 'is_matrix': False}
        pose_mat_step3 = build_pose_refine_quater(pose_mat_step2, init_roate_vec, refine_trpe='R')

        fwd_warp_parameter['pose'] = pose_mat_step3['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step3 = transformer_old(image2, depth_flow, [256, 832])

        # *****************************************************************************************************************
        '''旋转优化'''
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step3], axis=3)
        delta_roate_vec = {'pose': pose_net_loop_roate_refine_quater(posenet_inputs, is_training, variable_reuse), 'is_matrix': False}
        pose_mat_step4 = build_pose_refine_quater(pose_mat_step3, delta_roate_vec, refine_trpe='R')

        fwd_warp_parameter['pose'] = pose_mat_step4['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        # curr_proj_image1_depth_step4 = transformer_old(image2, depth_flow, [256, 832])
        # img_list = [(image1, curr_proj_image1_depth_step4)]
        return pose_mat_step3, pose_mat_step4

def build_posenet_quater_loop2(image1, image2, fwd_warp_parameter, bwd_warp_parameter, is_training=True,
                               variable_reuse=False):
    # *****************************************************************************************************************
    reuse_variables = False if is_training else True

    _, H, W, _ = image1.get_shape().as_list()
    with tf.variable_scope('pose_net_refine', reuse=reuse_variables):
        '''第一步'''
        init_pose = {'pose': fwd_warp_parameter['pose'], 'is_matrix': True}
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step1 = transformer_old(image2, depth_flow, [H, W])
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step1], axis=3)
        init_trans_vec = {'pose': pose_net_loop_trans(posenet_inputs, is_training, variable_reuse), 'is_matrix': False}
        pose_mat_step1 = build_pose_refine_quater(init_pose, init_trans_vec, refine_trpe='T')

        fwd_warp_parameter['pose'] = pose_mat_step1['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step1 = transformer_old(image2, depth_flow, [H, W])

        '''第二步'''
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step1], axis=3)
        delta_trans_vec = {'pose': pose_net_loop_trans_refine(posenet_inputs, is_training, variable_reuse),
                           'is_matrix': False}
        pose_mat_step2 = build_pose_refine_quater(pose_mat_step1, delta_trans_vec, refine_trpe='T')

        fwd_warp_parameter['pose'] = pose_mat_step2['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step2 = transformer_old(image2, depth_flow, [H, W])

        '''第三步'''
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step2], axis=3)
        init_roate_vec = {'pose': pose_net_loop_roate_quater(posenet_inputs, is_training, variable_reuse),
                          'is_matrix': False}
        pose_mat_step3 = build_pose_refine_quater(pose_mat_step2, init_roate_vec, refine_trpe='R')

        fwd_warp_parameter['pose'] = pose_mat_step3['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        curr_proj_image1_depth_step3 = transformer_old(image2, depth_flow, [H, W])

        '''第四步'''
        posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step3], axis=3)
        # posenet_inputs = tf.concat([image1, curr_proj_image1_depth_step3], axis=3)
        delta_roate_vec = {'pose': pose_net_loop_roate_refine_quater(posenet_inputs, is_training, variable_reuse),
                           'is_matrix': False}
        pose_mat_step4 = build_pose_refine_quater(pose_mat_step3, delta_roate_vec, refine_trpe='R')

        fwd_warp_parameter['pose'] = pose_mat_step4['pose']
        depth_flow, pose_mat = inverse_warp(
            **fwd_warp_parameter)
        # curr_proj_image1_depth_step4 = transformer_old(image2, depth_flow, [H, W])
        return pose_mat_step3, pose_mat_step4