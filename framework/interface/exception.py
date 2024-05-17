#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class RestartException(Exception):
    """
    标识用户希望重新开始一个新的episode
    """

    def __init__(self, client_id, ep_id, data):
        super().__init__()

        self._client_id = client_id
        self._ep_id = ep_id
        self._data = data

    @property
    def client_id(self):
        return self._client_id

    @property
    def ep_id(self):
        return self._ep_id

    @property
    def data(self):
        return self._data


class SkipEpisodeException(Exception):
    """
    标识只接收到了ep_start和ep_end消息
    """

    def __init__(self, client_id, ep_id):
        super().__init__()

        self._client_id = client_id
        self._ep_id = ep_id

    @property
    def client_id(self):
        return self._client_id

    @property
    def ep_id(self):
        return self._ep_id


class TimeoutEpisodeException(Exception):
    """
    标识episode超时
    """

    def __init__(self, client_id, ep_id):
        super().__init__()

        self._client_id = client_id
        self._ep_id = ep_id

    @property
    def client_id(self):
        return self._client_id

    @property
    def ep_id(self):
        return self._ep_id


class ClientQuitException(Exception):
    """
    标识客户端主动关闭连接
    """

    def __init__(self, client_id, quit_code, message):
        super().__init__()

        self._client_id = client_id
        self._quit_code = quit_code
        self._message = message

    @property
    def client_id(self):
        return self._client_id

    @property
    def quit_code(self):
        return self._quit_code

    @property
    def message(self):
        return self._message
