# automatically generated by the FlatBuffers compiler, do not modify

# namespace: kaiwu_msg

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class EpStartRsp(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = EpStartRsp()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsEpStartRsp(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # EpStartRsp
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # EpStartRsp
    def RetCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # EpStartRsp
    def EpId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # EpStartRsp
    def FrameInterval(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 1

def EpStartRspStart(builder): builder.StartObject(3)
def Start(builder):
    return EpStartRspStart(builder)
def EpStartRspAddRetCode(builder, retCode): builder.PrependInt32Slot(0, retCode, 0)
def AddRetCode(builder, retCode):
    return EpStartRspAddRetCode(builder, retCode)
def EpStartRspAddEpId(builder, epId): builder.PrependUint32Slot(1, epId, 0)
def AddEpId(builder, epId):
    return EpStartRspAddEpId(builder, epId)
def EpStartRspAddFrameInterval(builder, frameInterval): builder.PrependUint32Slot(2, frameInterval, 1)
def AddFrameInterval(builder, frameInterval):
    return EpStartRspAddFrameInterval(builder, frameInterval)
def EpStartRspEnd(builder): return builder.EndObject()
def End(builder):
    return EpStartRspEnd(builder)