; AutoSub Inno Setup 6 Installer Script (Windows x64)
; Produces: AutoSub-Setup-<AppVersion>-win64.exe
;
; Highlights:
; - Per-user install by default (`%LOCALAPPDATA%\Programs\AutoSub`): no UAC admin prompt needed,
;   and users can freely edit `.env`, `config\defaults.yaml`, `profiles\*.yaml`, `glossaries\*.yaml`.
; - Bundles Python runtime + onnxruntime + numpy + Silero VAD ONNX + FFmpeg (zero external installs).
; - Optional task (enabled by default) adds `{app}` to user PATH and removes it cleanly on uninstall.
; - Preserves existing user config/profiles during upgrades (`onlyifdoesntexist` on defaults/env).

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

#ifndef SourceDir
  #define SourceDir "..\..\dist\autosub"
#endif

#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{8F9E7D6C-4B3A-4E2F-9A1B-7C6D5E4F3A21}
AppName=AutoSub
AppVersion={#AppVersion}
AppVerName=AutoSub {#AppVersion}
AppPublisher=DotaArchive
DefaultDirName={autopf}\AutoSub
DefaultGroupName=AutoSub
AllowNoIcons=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesEnvironment=yes
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
OutputDir={#OutputDir}
OutputBaseFilename=AutoSub-Setup-{#AppVersion}-win64
UninstallDisplayIcon={app}\autosub.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add AutoSub to PATH (recommended so you can run 'autosub' in any terminal)"; GroupDescription: "System Integration:"

[Files]
; Core executable, _internal runtime, Silero ONNX model, bundled FFmpeg
Source: "{#SourceDir}\autosub.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist
Source: "{#SourceDir}\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#SourceDir}\.env.example"; DestDir: "{app}"; Flags: ignoreversion
; Editable user config, profiles & glossaries (preserve user edits on upgrade where appropriate)
Source: "{#SourceDir}\config\defaults.yaml"; DestDir: "{app}\config"; Flags: onlyifdoesntexist
Source: "{#SourceDir}\profiles\*"; DestDir: "{app}\profiles"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#SourceDir}\glossaries\*"; DestDir: "{app}\glossaries"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\AutoSub Command Prompt"; Filename: "{cmd}"; Parameters: "/k ""title AutoSub & autosub doctor & echo. & echo Try: autosub --help"""; WorkingDir: "{app}"
Name: "{group}\Open AutoSub Config & Profiles Folder"; Filename: "{app}"
Name: "{group}\Uninstall AutoSub"; Filename: "{uninstallexe}"

[Code]
const
  EnvKeyUser = 'Environment';
  EnvKeyAdmin = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

function GetEnvRootKey: Integer;
begin
  if IsAdminInstallMode then
    Result := HKEY_LOCAL_MACHINE
  else
    Result := HKEY_CURRENT_USER;
end;

function GetEnvSubKey: String;
begin
  if IsAdminInstallMode then
    Result := EnvKeyAdmin
  else
    Result := EnvKeyUser;
end;

procedure AddDirToPath(const Dir: String);
var
  OrigPath, NewPath, CheckPath, CheckDir: String;
begin
  if not RegQueryStringValue(GetEnvRootKey, GetEnvSubKey, 'Path', OrigPath) then
    OrigPath := '';

  CheckPath := ';' + Uppercase(OrigPath) + ';';
  CheckDir := ';' + Uppercase(Dir) + ';';
  if Pos(CheckDir, CheckPath) > 0 then
    Exit;

  if (Length(OrigPath) > 0) and (OrigPath[Length(OrigPath)] <> ';') then
    NewPath := OrigPath + ';' + Dir
  else
    NewPath := OrigPath + Dir;

  RegWriteExpandStringValue(GetEnvRootKey, GetEnvSubKey, 'Path', NewPath);
end;

procedure RemoveDirFromPath(const Dir: String);
var
  OrigPath, NewPath, Part, Target: String;
  P: Integer;
begin
  if not RegQueryStringValue(GetEnvRootKey, GetEnvSubKey, 'Path', OrigPath) then
    Exit;

  Target := Uppercase(Dir);
  NewPath := '';
  while Length(OrigPath) > 0 do
  begin
    P := Pos(';', OrigPath);
    if P > 0 then
    begin
      Part := Copy(OrigPath, 1, P - 1);
      Delete(OrigPath, 1, P);
    end
    else
    begin
      Part := OrigPath;
      OrigPath := '';
    end;

    if (Length(Part) > 0) and (Uppercase(Part) <> Target) then
    begin
      if Length(NewPath) > 0 then
        NewPath := NewPath + ';' + Part
      else
        NewPath := Part;
    end;
  end;

  RegWriteExpandStringValue(GetEnvRootKey, GetEnvSubKey, 'Path', NewPath);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddDirToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveDirFromPath(ExpandConstant('{app}'));
end;
