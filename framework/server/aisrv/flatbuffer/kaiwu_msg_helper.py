#!/usr/bin/env python
# -*- coding: utf-8 -*-


from framework.server.aisrv.flatbuffer.kaiwu_msg import *

class KaiwuMsgHelper:
    @staticmethod
    def encode_init_req(builder, client_id, client_version, data):
        client_id = builder.CreateString(client_id)
        client_version = builder.CreateString(client_version)
        data = builder.CreateByteVector(data)

        InitReq.InitReqStart(builder)
        InitReq.InitReqAddClientId(builder, client_id)
        InitReq.InitReqAddClientVersion(builder, client_version)
        InitReq.InitReqAddData(builder, data)
        req = InitReq.InitReqEnd(builder)

        return req

    @staticmethod
    def decode_init_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        client_version = str(req.ClientVersion(), encoding='utf8')
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, client_version, req_data

    @staticmethod
    def encode_init_rsp(builder, ret_code):
        InitRsp.InitRspStart(builder)
        InitRsp.InitRspAddRetCode(builder, ret_code)
        rsp = InitRsp.InitRspEnd(builder)

        return rsp

    @staticmethod
    def decode_init_rsp(rsp):
        ret_code = rsp.RetCode()

        return ret_code

    @staticmethod
    def encode_ep_start_req(builder, client_id, ep_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        EpStartReq.EpStartReqStart(builder)
        EpStartReq.EpStartReqAddClientId(builder, client_id)
        EpStartReq.EpStartReqAddEpId(builder, ep_id)
        EpStartReq.EpStartReqAddData(builder, data)
        req = EpStartReq.EpStartReqEnd(builder)

        return req

    @staticmethod
    def decode_ep_start_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        ep_id = req.EpId()
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, ep_id, req_data

    @staticmethod
    def encode_ep_start_rsp(builder, ret_code, ep_id, frame_interval):
        EpStartRsp.EpStartRspStart(builder)
        EpStartRsp.EpStartRspAddRetCode(builder, ret_code)
        EpStartRsp.EpStartRspAddEpId(builder, ep_id)
        EpStartRsp.EpStartRspAddFrameInterval(builder, frame_interval)
        rsp = EpStartRsp.EpStartRspEnd(builder)

        return rsp

    @staticmethod
    def decode_ep_start_rsp(rsp):
        ret_code = rsp.RetCode()
        ep_id = rsp.EpId()
        frame_interval = rsp.FrameInterval()

        return ret_code, ep_id, frame_interval

    @staticmethod
    def encode_agent_start_req(builder, client_id, ep_id, agent_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        AgentStartReq.AgentStartReqStart(builder)
        AgentStartReq.AgentStartReqAddClientId(builder, client_id)
        AgentStartReq.AgentStartReqAddEpId(builder, ep_id)
        AgentStartReq.AgentStartReqAddAgentId(builder, agent_id)
        AgentStartReq.AgentStartReqAddData(builder, data)
        req = AgentStartReq.AgentStartReqEnd(builder)

        return req

    @staticmethod
    def decode_agent_start_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        ep_id = req.EpId()
        agent_id = req.AgentId()
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, ep_id, agent_id, req_data

    @staticmethod
    def encode_agent_start_rsp(builder, ret_code, ep_id, agent_id):
        AgentStartRsp.AgentStartRspStart(builder)
        AgentStartRsp.AgentStartRspAddRetCode(builder, ret_code)
        AgentStartRsp.AgentStartRspAddEpId(builder, ep_id)
        AgentStartRsp.AgentStartRspAddAgentId(builder, agent_id)
        rsp = AgentStartRsp.AgentStartRspEnd(builder)

        return rsp

    @staticmethod
    def decode_agent_start_rsp(rsp):
        ret_code = rsp.RetCode()
        ep_id = rsp.EpId()
        agent_id = rsp.AgentId()

        return ret_code, ep_id, agent_id

    @staticmethod
    def encode_update_req_data(builder, data):
        update_req_data = []
        for agent_id, bytes_list in data.items():
            frames = []
            for byte_vec in bytes_list:
                byte_vec = builder.CreateByteVector(byte_vec)
                FrameData.FrameDataStart(builder)
                FrameData.FrameDataAddData(builder, byte_vec)
                frame = FrameData.FrameDataEnd(builder)
                frames.append(frame)

            UpdateReqData.UpdateReqDataStartFramesVector(builder, len(frames))
            for i in reversed(range(len(frames))):
                builder.PrependUOffsetTRelative(frames[i])
            frames = builder.EndVector(len(frames))

            UpdateReqData.Start(builder)
            UpdateReqData.AddAgentId(builder, agent_id)
            UpdateReqData.AddFrames(builder, frames)
            update_req_data.append(UpdateReqData.End(builder))
        return update_req_data
    
    @staticmethod
    def encode_update_req(builder, client_id, ep_id, data):

        update_req_data = KaiwuMsgHelper.encode_update_req_data(builder, data)

        UpdateReq.StartDataVector(builder, len(update_req_data))
        for i in reversed(range(len(update_req_data))):
            builder.PrependUOffsetTRelative(update_req_data[i])
        data = builder.EndVector(len(update_req_data))

        client_id = builder.CreateString(client_id)
        UpdateReq.UpdateReqStart(builder)
        UpdateReq.UpdateReqAddClientId(builder, client_id)
        UpdateReq.UpdateReqAddEpId(builder, ep_id)
        UpdateReq.UpdateReqAddData(builder, data)
        req = UpdateReq.UpdateReqEnd(builder)

        return req

    @staticmethod
    def decode_update_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        ep_id = req.EpId()
        data = {}
        for i in range(req.DataLength()):
            update_req_data = req.Data(i)
            agent_id = update_req_data.AgentId()
            frames = []
            for j in range(update_req_data.FramesLength()):
                frames.append(update_req_data.Frames(j).DataAsNumpy().tobytes())
            data[agent_id] = frames

        return client_id, ep_id, data

    @staticmethod
    def encode_update_rsp(builder, ret_code, ep_id, data):
        agent_ids, actions = [], []
        for agent_id, action in data.items():
            agent_ids.append(agent_id)
            actions.append(builder.CreateByteVector(action))
        update_rsp_data = []
        for agent_id, action in zip(agent_ids, actions):
            UpdateRspData.UpdateRspDataStart(builder)
            UpdateRspData.UpdateRspDataAddAgentId(builder, agent_id)
            UpdateRspData.UpdateRspDataAddAction(builder, action)
            data = UpdateRspData.UpdateRspDataEnd(builder)
            update_rsp_data.append(data)

        UpdateRsp.UpdateRspStartDataVector(builder, len(update_rsp_data))
        for i in reversed(range(len(update_rsp_data))):
            builder.PrependUOffsetTRelative(update_rsp_data[i])
        data = builder.EndVector(len(update_rsp_data))

        UpdateRsp.UpdateRspStart(builder)
        UpdateRsp.UpdateRspAddRetCode(builder, ret_code)
        UpdateRsp.UpdateRspAddEpId(builder, ep_id)
        UpdateRsp.UpdateRspAddData(builder, data)
        rsp = UpdateRsp.UpdateRspEnd(builder)

        return rsp

    @staticmethod
    def decode_update_rsp(rsp):
        ret_code = rsp.RetCode()
        ep_id = rsp.EpId()

        data = {}
        for i in range(rsp.DataLength()):
            update_rsp_data = rsp.Data(i)
            data[update_rsp_data.AgentId()] = update_rsp_data.ActionAsNumpy().tobytes()

        return ret_code, ep_id, data

    @staticmethod
    def encode_agent_end_req(builder, client_id, ep_id, agent_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        AgentEndReq.AgentEndReqStart(builder)
        AgentEndReq.AgentEndReqAddClientId(builder, client_id)
        AgentEndReq.AgentEndReqAddEpId(builder, ep_id)
        AgentEndReq.AgentEndReqAddAgentId(builder, agent_id)
        AgentEndReq.AgentEndReqAddData(builder, data)
        req = AgentEndReq.AgentEndReqEnd(builder)

        return req

    @staticmethod
    def decode_agent_end_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        ep_id = req.EpId()
        agent_id = req.AgentId()
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, ep_id, agent_id, req_data

    @staticmethod
    def encode_agent_end_rsp(builder, ret_code, ep_id, agent_id):
        AgentEndRsp.AgentEndRspStart(builder)
        AgentEndRsp.AgentEndRspAddRetCode(builder, ret_code)
        AgentEndRsp.AgentEndRspAddEpId(builder, ep_id)
        AgentEndRsp.AgentEndRspAddAgentId(builder, agent_id)
        rsp = AgentEndRsp.AgentEndRspEnd(builder)

        return rsp

    @staticmethod
    def decode_agent_end_rsp(rsp):
        ret_code = rsp.RetCode()
        ep_id = rsp.EpId()
        agent_id = rsp.AgentId()

        return ret_code, ep_id, agent_id

    @staticmethod
    def encode_ep_end_req(builder, client_id, ep_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        EpEndReq.EpEndReqStart(builder)
        EpEndReq.EpEndReqAddClientId(builder, client_id)
        EpEndReq.EpEndReqAddEpId(builder, ep_id)
        EpEndReq.EpEndReqAddData(builder, data)
        req = EpEndReq.EpEndReqEnd(builder)

        return req

    @staticmethod
    def decode_ep_end_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        ep_id = req.EpId()
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, ep_id, req_data

    @staticmethod
    def encode_ep_end_rsp(builder, ret_code, ep_id):
        EpEndRsp.EpEndRspStart(builder)
        EpEndRsp.EpEndRspAddRetCode(builder, ret_code)
        EpEndRsp.EpEndRspAddEpId(builder, ep_id)
        rsp = EpEndRsp.EpEndRspEnd(builder)

        return rsp

    @staticmethod
    def decode_ep_end_rsp(rsp):
        ret_code = rsp.RetCode()
        ep_id = rsp.EpId()

        return ret_code, ep_id

    @staticmethod
    def encode_event_req(builder, client_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        EventReq.EventReqStart(builder)
        EventReq.EventReqAddClientId(builder, client_id)
        EventReq.EventReqAddData(builder, data)
        req = EventReq.EventReqEnd(builder)

        return req

    @staticmethod
    def decode_event_req(req):
        client_id = str(req.ClientId(), encoding='utf8')
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, req_data

    @staticmethod
    def encode_event_rsp(builder, ret_code, data):
        data = builder.CreateByteVector(data)

        EventRsp.EventRspStart(builder)
        EventRsp.EventRspAddRetCode(builder, ret_code)
        EventRsp.EventRspAddData(builder, data)
        rsp = EventRsp.EventRspEnd(builder)

        return rsp

    @staticmethod
    def decode_event_rsp(rsp):
        ret_code = rsp.RetCode()
        data = rsp.DataAsNumpy().tobytes()

        return ret_code, data

    @staticmethod
    def encode_quit(builder, client_id, quit_code, message):
        client_id = builder.CreateString(client_id)
        message = builder.CreateString(message)

        Quit.QuitStart(builder)
        Quit.QuitAddClientId(builder, client_id)
        Quit.QuitAddQuitCode(builder, quit_code)
        Quit.QuitAddMessage(builder, message)
        req = Quit.QuitEnd(builder)

        return req

    @staticmethod
    def decode_quit(req):
        client_id = str(req.ClientId(), encoding='utf8')
        quit_code = req.QuitCode()
        message = str(req.Message(), encoding='utf8')

        return client_id, quit_code, message

    @staticmethod
    def encode_heartbeat(builder, client_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        HeartBeat.HeartBeatStart(builder)
        HeartBeat.HeartBeatAddClientId(builder, client_id)
        HeartBeat.HeartBeatAddData(builder, data)
        req = HeartBeat.HeartBeatEnd(builder)

        return req

    @staticmethod
    def decode_heartbeat(req):
        client_id = str(req.ClientId(), encoding='utf8')
        req_data = KaiwuMsgHelper.try_to_decode_data(req)

        return client_id, req_data

    @staticmethod
    def encode_restart(builder, client_id, ep_id, data):
        client_id = builder.CreateString(client_id)
        data = builder.CreateByteVector(data)

        Restart.RestartStart(builder)
        Restart.RestartAddClientId(builder, client_id)
        Restart.RestartAddEpId(builder, ep_id)
        Restart.RestartAddData(builder, data)
        rsp = Restart.RestartEnd(builder)

        return rsp

    @staticmethod
    def decode_restart(rsp):
        client_id = str(rsp.ClientId(), encoding='utf8')
        ep_id = rsp.EpId()
        data = rsp.DataAsNumpy().tobytes()

        return client_id, ep_id, data

    @staticmethod
    def encode_reject(builder, ret_code, message):
        message = builder.CreateString(message)

        Reject.RejectStart(builder)
        Reject.RejectAddRetCode(builder, ret_code)
        Reject.RejectAddMessage(builder, message)
        rsp = Reject.RejectEnd(builder)

        return rsp

    @staticmethod
    def decode_reject(rsp):
        ret_code = rsp.RetCode()
        message = str(rsp.Message(), encoding='utf8')

        return ret_code, message

    @staticmethod
    def encode_request(builder, seq_no, msg_type, msg):
        Request.RequestStart(builder)
        Request.RequestAddSeqNo(builder, seq_no)
        Request.RequestAddMsgType(builder, msg_type)
        Request.RequestAddMsg(builder, msg)
        req = Request.RequestEnd(builder)
        return req

    @staticmethod
    def try_to_decode_data(req):
        """ try to decode self-defined data field in the req as byte array
        Args:
            req (KaiwuXXReq): req which has data field
        Returns (): byte array if succeeded else None
        """
        try:
            return req.DataAsNumpy().tobytes()
        except Exception: 
            return None

    @staticmethod
    def decode_request(req):
        seq_no = req.SeqNo()

        if req.MsgType() == ReqMsg.ReqMsg.init_req:
            msg = InitReq.InitReq()
        elif req.MsgType() == ReqMsg.ReqMsg.ep_start_req:
            msg = EpStartReq.EpStartReq()
        elif req.MsgType() == ReqMsg.ReqMsg.agent_start_req:
            msg = AgentStartReq.AgentStartReq()
        elif req.MsgType() == ReqMsg.ReqMsg.update_req:
            msg = UpdateReq.UpdateReq()
        elif req.MsgType() == ReqMsg.ReqMsg.agent_end_req:
            msg = AgentEndReq.AgentEndReq()
        elif req.MsgType() == ReqMsg.ReqMsg.ep_end_req:
            msg = EpEndReq.EpEndReq()
        elif req.MsgType() == ReqMsg.ReqMsg.quit:
            msg = Quit.Quit()
        elif req.MsgType() == ReqMsg.ReqMsg.event_req:
            msg = EventReq.EventReq()
        elif req.MsgType() == ReqMsg.ReqMsg.heartbeat:
            msg = HeartBeat.HeartBeat()
        else:
            raise RuntimeError("illegal request msg type %d" % req.MsgType())

        msg.Init(req.Msg().Bytes, req.Msg().Pos)
        return seq_no, req.MsgType(), msg

    @staticmethod
    def encode_response(builder, seq_no, msg_type, msg):
        Response.ResponseStart(builder)
        Response.ResponseAddSeqNo(builder, seq_no)
        Response.ResponseAddMsgType(builder, msg_type)
        Response.ResponseAddMsg(builder, msg)
        rsp = Response.ResponseEnd(builder)

        return rsp

    @staticmethod
    def decode_response(rsp):
        seq_no = rsp.SeqNo()

        if rsp.MsgType() == RspMsg.RspMsg.init_rsp:
            msg = InitRsp.InitRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.ep_start_rsp:
            msg = EpStartRsp.EpStartRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.agent_start_rsp:
            msg = AgentStartRsp.AgentStartRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.update_rsp:
            msg = UpdateRsp.UpdateRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.agent_end_rsp:
            msg = AgentEndRsp.AgentEndRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.ep_end_rsp:
            msg = EpEndRsp.EpEndRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.event_rsp:
            msg = EventRsp.EventRsp()
        elif rsp.MsgType() == RspMsg.RspMsg.restart:
            msg = Restart.Restart()
        elif rsp.MsgType() == RspMsg.RspMsg.reject:
            msg = Reject.Reject()
        else:
            raise RuntimeError("illegal responese msg type %d" % rsp.MsgType())

        msg.Init(rsp.Msg().Bytes, rsp.Msg().Pos)
        return seq_no, rsp.MsgType(), msg
