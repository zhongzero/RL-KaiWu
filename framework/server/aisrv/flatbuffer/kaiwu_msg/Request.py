# automatically generated by the FlatBuffers compiler, do not modify

# namespace: kaiwu_msg

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Request(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Request()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRequest(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Request
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Request
    def SeqNo(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Request
    def MsgType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint8Flags, o + self._tab.Pos)
        return 0

    # Request
    def Msg(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            from flatbuffers.table import Table
            obj = Table(bytearray(), 0)
            self._tab.Union(obj, o)
            return obj
        return None

def RequestStart(builder): builder.StartObject(3)
def Start(builder):
    return RequestStart(builder)
def RequestAddSeqNo(builder, seqNo): builder.PrependUint32Slot(0, seqNo, 0)
def AddSeqNo(builder, seqNo):
    return RequestAddSeqNo(builder, seqNo)
def RequestAddMsgType(builder, msgType): builder.PrependUint8Slot(1, msgType, 0)
def AddMsgType(builder, msgType):
    return RequestAddMsgType(builder, msgType)
def RequestAddMsg(builder, msg): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(msg), 0)
def AddMsg(builder, msg):
    return RequestAddMsg(builder, msg)
def RequestEnd(builder): return builder.EndObject()
def End(builder):
    return RequestEnd(builder)