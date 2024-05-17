# automatically generated by the FlatBuffers compiler, do not modify

# namespace: kaiwu_msg

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class UpdateReq(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UpdateReq()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsUpdateReq(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # UpdateReq
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UpdateReq
    def ClientId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # UpdateReq
    def EpId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # UpdateReq
    def Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .UpdateReqData import UpdateReqData
            obj = UpdateReqData()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # UpdateReq
    def DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # UpdateReq
    def DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

def UpdateReqStart(builder): builder.StartObject(3)
def Start(builder):
    return UpdateReqStart(builder)
def UpdateReqAddClientId(builder, clientId): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(clientId), 0)
def AddClientId(builder, clientId):
    return UpdateReqAddClientId(builder, clientId)
def UpdateReqAddEpId(builder, epId): builder.PrependUint32Slot(1, epId, 0)
def AddEpId(builder, epId):
    return UpdateReqAddEpId(builder, epId)
def UpdateReqAddData(builder, data): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)
def AddData(builder, data):
    return UpdateReqAddData(builder, data)
def UpdateReqStartDataVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartDataVector(builder, numElems):
    return UpdateReqStartDataVector(builder, numElems)
def UpdateReqEnd(builder): return builder.EndObject()
def End(builder):
    return UpdateReqEnd(builder)