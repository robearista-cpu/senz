' ===========================================================================
' senz_hub.vbs -- flash-free double-click launcher for the senz control hub.
'
' Runs senz_hub.bat with a hidden window, so the hub opens with NO console
' window at all (feels like a normal app). The .bat does the Python lookup.
' ===========================================================================
Dim fso, sh, here
Set fso = CreateObject("Scripting.FileSystemObject")
Set sh = CreateObject("WScript.Shell")
here = fso.GetParentFolderName(WScript.ScriptFullName)
' 0 = hidden window, False = don't wait for it to finish.
sh.Run """" & here & "\senz_hub.bat""", 0, False
