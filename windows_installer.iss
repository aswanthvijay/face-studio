#ifndef AppVersion
  #define AppVersion "1.0.0-dev"
#endif

[Setup]
AppId={{8E2D592A-12C1-43DD-AEA7-9EF0C4370B17}
AppName=Face Studio
AppVersion={#AppVersion}
AppPublisher=Face Studio
DefaultDirName={localappdata}\Programs\FaceStudio
DefaultGroupName=Face Studio
UninstallDisplayName=Face Studio
UninstallDisplayIcon={app}\FaceStudio.exe
OutputDir=dist\installer
OutputBaseFilename=FaceStudio-Setup
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
Compression=lzma2
SolidCompression=yes
CloseApplications=yes
RestartApplications=yes
DisableProgramGroupPage=yes
SetupLogging=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "dist\windows\FaceStudio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Face Studio"; Filename: "{app}\FaceStudio.exe"; WorkingDir: "{app}"
Name: "{autodesktop}\Face Studio"; Filename: "{app}\FaceStudio.exe"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\FaceStudio.exe"; Description: "Launch Face Studio"; Flags: nowait postinstall skipifsilent
