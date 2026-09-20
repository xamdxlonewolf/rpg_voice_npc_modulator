; Copyright (C) 2026 Michael Cobb
; SPDX-License-Identifier: GPL-3.0-or-later
; Per-user Inno Setup installer. Version is stamped from pyproject.toml.

#define MyAppName "Voice of the Realm"
#define MyAppPublisher "Michael Cobb"
#define MyAppExeName "VoiceOfTheRealm.exe"
#define MyAppURL "https://github.com/xamdxlonewolf/rpg_voice_npc_modulator"

#ifndef MyAppVersion
#include "version.iss"
#endif

[Setup]
AppId={{8F3C2A91-4B67-4E2D-9C11-7A1B2C3D4E5F}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={localappdata}\VoiceOfTheRealm
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
LicenseFile=..\LICENSE
InfoAfterFile=..\THIRD_PARTY_NOTICES.md
OutputDir=output
OutputBaseFilename=VoiceOfTheRealm-{#MyAppVersion}-setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
SetupIconFile=
; Virtual Cable drivers are not bundled (ADR-0006).

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\dist\VoiceOfTheRealm\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
