#!/usr/bin/env python3
# -*- coding:utf-8 -*-


from __future__ import print_function
import collections as pycoll
import operator
from framework.common.utils.tf_utils import *

AutoLossScaleParams = pycoll.namedtuple(
    'AutoLossScaleParams',
    [
        'enable_auto_loss_scale',
        'loss_scale',
        'loss_scale_normal_steps',
        'inc_loss_scale_every_n',
        'increase_loss_scale',
        'decrease_loss_scale',
    ])

class VariableMgrIndependent(object):
  def __init__(self, benchmark_cnn):
    self.benchmark_cnn = benchmark_cnn
    self.staging_delta_ops = []
    # A variable for automatic loss scaling.
    self.grad_has_inf_nan = None


  def append_apply_gradients_ops(self, gradient_state, opt, grads, training_ops,
                                 loss_scale_params):
    del gradient_state  # unused by this implementation

    def get_apply_gradients_ops_func():
      return [opt.apply_gradients(grads)]

    loss_scale_update = append_gradients_with_loss_scale(
        training_ops, get_apply_gradients_ops_func, loss_scale_params,
        self.grad_has_inf_nan)
    return self.grad_has_inf_nan, loss_scale_update

  def savable_variables(self):
    params = []
    for v in tf.global_variables():
      #if not v.name.startswith('loss_scale') and not v.name.startswith('cond_if_grad_has_inf_nan'):
      if not v.name.startswith('input_datas')\
        and not v.name.startswith('cond_1/beta1_power')\
        and not v.name.startswith('cond_1/beta2_power')\
        and not v.name.startswith('good_steps')\
        and not v.name.startswith('current_loss_scale'):
        params.append(v)
    return params

  def trainable_variables_on_device(self, writable=False):
    del writable
    params = [
          v for v in tf.trainable_variables()
          if v.name.startswith('v0/')
    ]
    return params

  def get_gradients_to_apply(self, gradient_state):
    tower_grad = gradient_state

    if self.benchmark_cnn.enable_auto_loss_scale:
      has_inf_nan_list = []
      for grad, _ in tower_grad:
        has_inf_nan_list.append(tf.reduce_all(tf.is_finite(grad)))
      self.grad_has_inf_nan = tf.logical_not(tf.reduce_all(has_inf_nan_list))
    return tower_grad


def get_loss_scale_update_op(loss_scale, loss_scale_normal_steps,
                             inc_loss_scale_every_n, increase_loss_scale):
  def increment_loss_scale_normal_steps_func():
    return tf.group(loss_scale_normal_steps.assign_add(1))

  def increase_loss_scale_func():
    new_loss_scale = tf.cond(tf.is_finite(loss_scale * increase_loss_scale),
        lambda: loss_scale * increase_loss_scale,
        lambda: loss_scale)
    return tf.group(
        tf.assign(loss_scale_normal_steps, 0),
        tf.assign(loss_scale, new_loss_scale))

  return tf.cond(loss_scale_normal_steps < inc_loss_scale_every_n,
                 increment_loss_scale_normal_steps_func,
                 increase_loss_scale_func)


def append_gradients_with_loss_scale(training_ops, get_apply_gradients_ops_func,
                                     loss_scale_params, grad_has_inf_nan):
  loss_scale = loss_scale_params.loss_scale
  loss_scale_normal_steps = loss_scale_params.loss_scale_normal_steps
  inc_loss_scale_every_n = loss_scale_params.inc_loss_scale_every_n
  enable_auto_loss_scale = loss_scale_params.enable_auto_loss_scale
  increase_loss_scale = loss_scale_params.increase_loss_scale
  decrease_loss_scale = loss_scale_params.decrease_loss_scale

  if loss_scale is None or not enable_auto_loss_scale:
    training_ops.extend(get_apply_gradients_ops_func())
  else:
    def update_op_if_nan_or_inf():
      return tf.group(
          tf.assign(loss_scale, tf.maximum(1., loss_scale * decrease_loss_scale)),
          tf.assign(loss_scale_normal_steps, 0))


    def update_op_if_no_nan_or_inf():
      return tf.group(
          get_loss_scale_update_op(loss_scale, loss_scale_normal_steps,
                                   inc_loss_scale_every_n, increase_loss_scale),
          *get_apply_gradients_ops_func())

    assert grad_has_inf_nan is not None
    update_op = tf.cond(
        grad_has_inf_nan,
        update_op_if_nan_or_inf,
        update_op_if_no_nan_or_inf,
        name='cond_if_grad_has_inf_nan'
    )
    training_ops.append(update_op)
  return loss_scale



