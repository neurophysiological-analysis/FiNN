'''
Created on Jul 22, 2025

@author: voodoocode
'''

def main():
    """
    This script is run *within* 3dslicer. As the imports only exisit for the 
    3d slicer built-in python environment, imports are part of the function call.
    """
    import sys
    import slicer
    import SegmentEditor
    import vtk
    
    print(sys.argv[1])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])
    print(sys.argv[5])
    
    in_path = str(sys.argv[1])
    out_path = str(sys.argv[2])
    
    seg_name = str(sys.argv[3])
         
    vol = slicer.util.loadVolume(in_path)
      
    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.CreateDefaultDisplayNodes()
    segmentationNode.GetSegmentation().AddEmptySegment(seg_name)
    segmentationNode.SetName(seg_name)
    segmentationNode.GetSegmentation().GetNthSegment(0).SetName(seg_name)
      
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorNode.SetAndObserveSegmentationNode(segmentationNode)
    segmentEditorNode.SetAndObserveSourceVolumeNode(vol)
      
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
      
    # Select or create a segment
    segmentID = segmentationNode.GetSegmentation().GetNthSegmentID(0)  # Use existing segment
    segmentEditorNode.SetSelectedSegmentID(segmentID)
    
    # Set and apply the Threshold effect
    segmentEditorWidget.setActiveEffectByName("Threshold")
    effect = segmentEditorWidget.activeEffect()
    effect.setParameter("MinimumThreshold", float(sys.argv[4]))
    effect.setParameter("MaximumThreshold", float(sys.argv[5]))
    effect.self().onApply()
      
    arr = vtk.vtkStringArray()
    arr.InsertNextValue(segmentID)
    slicer.modules.segmentations.logic().ExportSegmentsClosedSurfaceRepresentationToFiles(out_path, segmentationNode, arr, "obj")
    
    slicer.app.exit()

if __name__ == "__main__":
    main()
