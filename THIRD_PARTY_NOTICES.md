# Third-party notices

This repository is licensed under the MIT License (see `LICENSE`). Parts of it
are derived from third-party work under other licenses. This file lists those
parts, their origin and their license, and carries the license texts those
licenses require to be distributed with the derived work.

The project is not affiliated with or endorsed by NVIDIA, Hugging Face,
TheRobotStudio, LycheeAI or Lior Ben Horin. Product names are used only to
identify the software and hardware this project works with.

> **Status:** this file is being completed. The open items are listed under
> [Open items](#open-items) at the end. Until they are done, some source files
> still carry an `SPDX-License-Identifier: MIT` header although they are derived
> from the work listed below.

---

## 1. lerobot_so101_teleop — MIT

- **Source:** <https://github.com/liorbenhorin/lerobot_so101_teleop>
- **Copyright:** Copyright 2025 Lior Ben Horin
- **License:** MIT (full text in [Appendix A](#appendix-a--mit-license-lerobot_so101_teleop))
- **Used in this repository:**
  - `src/so101_mvbench/vendor/lerobot_so101_teleop/` — vendored code
    (upstream `LICENSE` kept in that directory)
  - `src/so101_mvbench/tasks/lift_cube_env_cfg.py` — based on upstream
    `tasks/base/base_env_cfg.py`, modified
  - `src/so101_mvbench/assets/so101_white.py` — actuator parameters copied from
    upstream `assets/so101.py`
  - `src/so101_mvbench/mdp/resets.py` — `randomize_light_exposure` derived from
    upstream `mdp/funcs.py`, modified for multi-env use
  - `src/so101_mvbench/assets/usd/SO-ARM101-USD-white-multienv.usd` — derived
    from upstream `assets/usd/SO-ARM101-USD.usd`, recoloured white and with the
    root joint removed (see also section 4)
  - `src/so101_mvbench/assets/usd/action-pad-plain.usda` — override of upstream
    `assets/usd/action-pad.usda` (logo and gloss removed)

## 2. Sim-to-Real-SO-101-Workshop — Apache-2.0

- **Source:** <https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop>
- **Copyright:** Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- **License:** Apache License 2.0 (full text in [Appendix C](#appendix-c--apache-license-20))
- **Used in this repository:**
  - `src/so101_mvbench/tools/list_envs.py` — adapted from
    `source/sim_to_real_so101/scripts/list_envs.py`, modified (package filter,
    plain-text table, hard exit)
  - `src/so101_mvbench/mdp/resets.py` — `randomize_robot_color` follows
    `source/sim_to_real_so101/mdp/resets.py` (same shader path and approach),
    rewritten for a discrete palette and multi-env use

## 3. Isaac Lab — BSD-3-Clause

- **Source:** <https://github.com/isaac-sim/IsaacLab>
- **Copyright:** Copyright (c) 2022-2025, The Isaac Lab Project Developers
  (<https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md>). All rights reserved.
- **License:** BSD-3-Clause (full text in [Appendix B](#appendix-b--bsd-3-clause-license-isaac-lab))
- **Used in this repository:**
  - `src/so101_mvbench/tools/list_envs.py` — the upstream workshop file this is
    adapted from itself contains code derived from Isaac Lab

## 4. SO-ARM100 / SO-ARM101 robot model — Apache-2.0

- **Source:** <https://github.com/TheRobotStudio/SO-ARM100>
- **Authors:** TheRobotStudio (Rob Knight), Pepijn Kooijmans, Remi Cadene, Simon Alibert
  and contributors ("Standard Open SO-100 & SO-101 Arms")
- **License:** Apache License 2.0 (full text in [Appendix C](#appendix-c--apache-license-20))
- **Used in this repository:** the geometry of the SO-ARM101 in
  `src/so101_mvbench/assets/usd/SO-ARM101-USD-white-multienv.usd`, and therefore
  every render of the arm in `docs/_static/` and in the published dataset.

**Provenance chain of the robot USD:**

1. TheRobotStudio SO-ARM100 — CAD / URDF, Apache-2.0
2. LycheeAI (Muammer Bay) — USD conversion `IsaacSim_USD/SO-ARM101-USD.usd` in
   <https://github.com/MuammerBay/so-arm101-ros2-bridge>. **That repository
   states no license.**
3. Lior Ben Horin — redistributed the USD in lerobot_so101_teleop (MIT)
4. This repository — recoloured, root joint removed

Step 2 is the one link without an explicit license; see [Open items](#open-items).

## 5. Not redistributed

These are dependencies, not part of this repository. They are installed or
loaded separately and are subject to their own licenses:

- NVIDIA Isaac Sim and Isaac Lab — installed from NVIDIA's package index; Isaac
  Sim requires accepting NVIDIA's license agreement on first launch
- LeRobot (Hugging Face, Apache-2.0) and the other Python dependencies in
  `pyproject.toml`
- three.js and its OrbitControls (MIT), loaded from a CDN by
  `docs/_static/hemisphere/hemisphere_viewer.html`

## 6. Original to this repository

Covered by this repository's MIT `LICENSE`:

- `src/so101_mvbench/assets/usd/cylindrical_room_shell.usda`,
  `action-pad-round.usda`, `action-pad-none.usda`
- `src/so101_mvbench/assets/usd/textures/pad-plain.png`, `pad-rough.png`
  (single-colour 64×64 fills)
- all images, videos and diagrams under `docs/_static/`

---

## Open items

To do before this file can be considered complete. Tick them off here as they
are done.

- [ ] **`tools/list_envs.py`** — restore the upstream headers: NVIDIA copyright
  plus `SPDX-License-Identifier: Apache-2.0`, the Isaac Lab BSD-3-Clause
  attribution, and a line stating that the file was modified. Replace the MIT
  SPDX header. Remove the reference to `.claude/memory/...` from the docstring.
- [ ] **`mdp/resets.py`** — add attribution comments to `randomize_light_exposure`
  (lerobot_so101_teleop, MIT) and `randomize_robot_color`
  (Sim-to-Real-SO-101-Workshop, Apache-2.0); mark the file as containing
  modified third-party code.
- [ ] **`tasks/lift_cube_env_cfg.py`, `assets/so101_white.py`** — add
  "Copyright 2025 Lior Ben Horin (MIT), see THIRD_PARTY_NOTICES.md" next to the
  existing "Based on" comments.
- [ ] **`tasks/__init__.py`** — check whether the gym registration boilerplate
  follows the workshop's `tasks/__init__.py`, which is itself derived from
  Isaac Lab; if so, attribute it like `list_envs.py`.
- [ ] **Robot USD provenance** — either regenerate the SO-ARM101 USD from
  TheRobotStudio's official URDF (Apache-2.0) with the Isaac Sim URDF importer,
  or ask LycheeAI / Muammer Bay for permission to redistribute their USD
  conversion. Then update section 4.
- [ ] **`README.md`, section "License and provenance"** — correct the robot
  USD chain (not simply "lerobot_so101_teleop, MIT"), correct "the room and pad
  USDs are original" (`action-pad-plain.usda` is derived), link to this file,
  and move the "Not affiliated …" line there.
- [ ] **`src/so101_mvbench/assets/usd/README.md`** — quote the MIT license in
  full (the warranty disclaimer is missing).
- [ ] **`action-pad-plain.usda`** — it references `./action-pad.usda`, which is
  not in the repository, so the file does not load. Delete it together with
  the `plain` option in `recording/multicam_replay.py`, or ship the upstream
  pad with its MIT notice.
- [ ] **`vendor/gamepad_utils/`** — if this is original code, move it out of
  `vendor/` or state in `vendor/__init__.py` that it is first-party; add a
  header to `motion.py`, which has none.
- [ ] **`vendor/__init__.py`** — points to `THIRD_PARTY_NOTICES`; change it to
  `THIRD_PARTY_NOTICES.md`.
- [ ] **`pyproject.toml`** — add `THIRD_PARTY_NOTICES.md` to `license-files` so
  it ships in the wheel.

---

## Appendix A — MIT License (lerobot_so101_teleop)

```text
Copyright 2025 Lior Ben Horin

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

## Appendix B — BSD-3-Clause License (Isaac Lab)

```text
Copyright (c) 2022-2025, The Isaac Lab Project Developers
(https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Appendix C — Apache License 2.0

Applies to the Sim-to-Real-SO-101-Workshop code (section 2) and the SO-ARM100
robot model (section 4).

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```
