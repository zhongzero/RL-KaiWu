# automatically generated by the FlatBuffers compiler, do not modify

# namespace: kaiwu_msg

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class UpdateRsp(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = UpdateRsp()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsUpdateRsp(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # UpdateRsp
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # UpdateRsp
    def RetCode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # UpdateRsp
    def EpId(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # UpdateRsp
    def Data(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from .UpdateRspData import UpdateRspData
            obj = UpdateRspData()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # UpdateRsp
    def DataLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # UpdateRsp
    def DataIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

def UpdateRspStart(builder): builder.StartObject(3)
def Start(builder):
    return UpdateRspStart(builder)
def UpdateRspAddRetCode(builder, retCode): builder.PrependInt32Slot(0, retCode, 0)
def AddRetCode(builder, retCode):
    return UpdateRspAddRetCode(builder, retCode)
def UpdateRspAddEpId(builder, epId): builder.PrependUint32Slot(1, epId, 0)
def AddEpId(builder, epId):
    return UpdateRspAddEpId(builder, epId)
def UpdateRspAddData(builder, data): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(data), 0)
def AddData(builder, data):
    return UpdateRspAddData(builder, data)
def UpdateRspStartDataVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def StartDataVector(builder, numElems):
    return UpdateRspStartDataVector(builder, numElems)
def UpdateRspEnd(builder): return builder.EndObject()
def End(builder):
    return UpdateRspEnd(builder)