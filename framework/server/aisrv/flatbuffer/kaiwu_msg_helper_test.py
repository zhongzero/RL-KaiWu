#!/usr/bin/env python
# -*- coding: utf-8 -*-


import unittest
import flatbuffers

from framework.server.aisrv.flatbuffer.kaiwu_msg import *
from framework.server.aisrv.flatbuffer.kaiwu_msg_helper import KaiwuMsgHelper

class TestKaiwuMsgHelper(unittest.TestCase):
    def test_init_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_init_req(builder, 'client_id', 'client_version', b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = InitReq.InitReq.GetRootAsInitReq(buf, 0)

        client_id, client_version, data = KaiwuMsgHelper.decode_init_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(client_version, "client_version")
        self.assertEqual(data, b'Hello World!')

    def test_init_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_init_rsp(builder, 0)

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = InitRsp.InitRsp.GetRootAsInitRsp(buf, 0)

        ret_code = KaiwuMsgHelper.decode_init_rsp(rsp)

        self.assertEqual(ret_code, 0)

    def test_ep_start_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_ep_end_req(builder, "client_id", 1, b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = EpStartReq.EpStartReq.GetRootAsEpStartReq(buf, 0)

        client_id, ep_id, data = KaiwuMsgHelper.decode_ep_start_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(ep_id, 1)
        self.assertEqual(data, b'Hello World!')

    def test_ep_start_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_ep_start_rsp(builder, 0, 1, 2)

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = EpStartRsp.EpStartRsp.GetRootAsEpStartRsp(buf, 0)

        ret_code, ep_id, frame_interval = KaiwuMsgHelper.decode_ep_start_rsp(rsp)

        self.assertEqual(ret_code, 0)
        self.assertEqual(ep_id, 1)
        self.assertEqual(frame_interval, 2)

    def test_agent_start_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_agent_start_req(builder, "client_id", 1, 2, b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = AgentStartReq.AgentStartReq.GetRootAsAgentStartReq(buf, 0)

        client_id, ep_id, agent_id, data = KaiwuMsgHelper.decode_agent_start_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(ep_id, 1)
        self.assertEqual(agent_id, 2)
        self.assertEqual(data, b'Hello World!')

    def test_agent_start_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_agent_start_rsp(builder, 0, 1, 2)

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = AgentStartRsp.AgentStartRsp.GetRootAsAgentStartRsp(buf, 0)

        ret_code, ep_id, agent_id = KaiwuMsgHelper.decode_agent_start_rsp(rsp)

        self.assertEqual(ret_code, 0)
        self.assertEqual(ep_id, 1)
        self.assertEqual(agent_id, 2)

    def test_update_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_update_req(builder, 'client_id', 0, {
            1: [b'Hello 1'],
            2: [b'Hello 2']
        })

        builder.Finish(req)

        buf = builder.Output()

        req = UpdateReq.UpdateReq.GetRootAsUpdateReq(buf, 0)

        client_id, ep_id, data = KaiwuMsgHelper.decode_update_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(ep_id, 0)

        for i, agent_id in enumerate(sorted(data)):
            self.assertEqual(agent_id, i + 1)
            frames = data[agent_id]
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0], bytes(f"Hello {i + 1}", encoding="utf8"))

    def test_update_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_update_rsp(builder, 0, 1, {
            1: b'act_1',
            2: b'act_2'
        })

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = UpdateRsp.UpdateRsp.GetRootAsUpdateRsp(buf, 0)

        self.assertEqual(rsp.RetCode(), 0)
        self.assertEqual(rsp.EpId(), 1)
        self.assertEqual(rsp.DataLength(), 2)

        for i in range(rsp.DataLength()):
            rsp_data = rsp.Data(i)
            self.assertEqual(rsp_data.AgentId(), i + 1)
            self.assertEqual(str(rsp_data.ActionAsNumpy().tobytes(), encoding='utf8'), f"act_{i + 1}")

    def test_agent_end_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_agent_end_req(builder, "client_id", 0, 1, b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = AgentEndReq.AgentEndReq.GetRootAsAgentEndReq(buf, 0)

        client_id, ep_id, agent_id, data = KaiwuMsgHelper.decode_agent_end_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(ep_id, 0)
        self.assertEqual(agent_id, 1)
        self.assertEqual(data, b'Hello World!')

    def test_agent_end_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_agent_end_rsp(builder, 0, 1, 2)

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = AgentEndRsp.AgentEndRsp.GetRootAsAgentEndRsp(buf, 0)

        ret_code, ep_id, agent_id = KaiwuMsgHelper.decode_agent_end_rsp(rsp)

        self.assertEqual(ret_code, 0)
        self.assertEqual(ep_id, 1)
        self.assertEqual(agent_id, 2)

    def test_ep_end_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_ep_end_req(builder, "client_id", 0, b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = EpEndReq.EpEndReq.GetRootAsEpEndReq(buf, 0)

        client_id, ep_id, data = KaiwuMsgHelper.decode_ep_end_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(ep_id, 0)
        self.assertEqual(data, b'Hello World!')

    def test_encode_ep_end_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_ep_end_rsp(builder, 0, 1)

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = EpEndRsp.EpEndRsp.GetRootAsEpEndRsp(buf, 0)

        ret_code, ep_id = KaiwuMsgHelper.decode_ep_end_rsp(rsp)

        self.assertEqual(ret_code, 0)
        self.assertEqual(ep_id, 1)

    def test_event_req(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_event_req(builder, 'client_id', b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = EventReq.EventReq.GetRootAsEventReq(buf, 0)

        client_id, data = KaiwuMsgHelper.decode_event_req(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(data, b'Hello World!')

    def test_event_rsp(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_event_rsp(builder, 0, b'Hello World')

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = EventRsp.EventRsp.GetRootAsEventRsp(buf, 0)

        ret_code, data = KaiwuMsgHelper.decode_event_rsp(rsp)

        self.assertEqual(ret_code, 0)
        self.assertEqual(data, b'Hello World')

    def test_quit(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_quit(builder, 'client_id', 0, 'message')

        builder.Finish(req)

        buf = builder.Output()

        req = Quit.Quit.GetRootAsQuit(buf, 0)

        client_id, quit_code, message = KaiwuMsgHelper.decode_quit(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(quit_code, 0)
        self.assertEqual(message, "message")

    def test_heartbeat(self):
        builder = flatbuffers.Builder(0)

        req = KaiwuMsgHelper.encode_heartbeat(builder, 'client_id', b'Hello World!')

        builder.Finish(req)

        buf = builder.Output()

        req = HeartBeat.HeartBeat.GetRootAsHeartBeat(buf, 0)

        client_id, data = KaiwuMsgHelper.decode_heartbeat(req)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(data, b'Hello World!')

    def test_restart(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_restart(builder, "client_id", 0, b'Hello World')

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = Restart.Restart.GetRootAsRestart(buf, 0)

        client_id, ep_id, data = KaiwuMsgHelper.decode_restart(rsp)

        self.assertEqual(client_id, "client_id")
        self.assertEqual(ep_id, 0)
        self.assertEqual(data, b'Hello World')

    def test_reject(self):
        builder = flatbuffers.Builder(0)

        rsp = KaiwuMsgHelper.encode_reject(builder, 0, 'Hello World')

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = Reject.Reject.GetRootAsReject(buf, 0)

        ret_code, message = KaiwuMsgHelper.decode_reject(rsp)

        self.assertEqual(ret_code, 0)
        self.assertEqual(message, 'Hello World')

    def test_request(self):
        builder = flatbuffers.Builder(0)

        init_req = KaiwuMsgHelper.encode_init_req(builder, 'client_id', 'client_version', b'Hello World!')

        req = KaiwuMsgHelper.encode_request(builder, 0, ReqMsg.ReqMsg.init_req, init_req)

        builder.Finish(req)

        buf = builder.Output()

        req = Request.Request.GetRootAsRequest(buf, 0)

        seq_no, msg_type, __ = KaiwuMsgHelper.decode_request(req)

        self.assertEqual(seq_no, 0)
        self.assertEqual(msg_type, ReqMsg.ReqMsg.init_req)

    def test_response(self):
        builder = flatbuffers.Builder(0)

        init_rsp = KaiwuMsgHelper.encode_init_rsp(builder, 0)

        rsp = KaiwuMsgHelper.encode_response(builder, 0, RspMsg.RspMsg.init_rsp, init_rsp)

        builder.Finish(rsp)

        buf = builder.Output()

        rsp = Response.Response.GetRootAsResponse(buf, 0)

        seq_no, msg_type, __ = KaiwuMsgHelper.decode_response(rsp)

        self.assertEqual(seq_no, 0)
        self.assertEqual(msg_type, RspMsg.RspMsg.init_rsp)


if __name__ == '__main__':
    unittest.main()
