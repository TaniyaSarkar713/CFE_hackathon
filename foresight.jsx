import { useState, useEffect } from "react";

const PAPER_DATA = {
  id: "11812686",
  title: "Multiagency approaches to preventing sudden unexpected death in infancy (SUDI): a review and analysis of UK policies",
  authors: "Helen L Ball, Alice-Amber Keegan, Daniel R Whitehouse, Louise S Cooper, Sophie R Lovell-Kennedy, Laura M Murray, Dorothy Newbury-Birch, Nicola J Cleghorn, Amanda Healy",
};

const SUMMARY = {
  topic_and_scope: "This analysis examines how English local authorities implement multiagency working (MAW) for SUDI prevention, highlighting policy variability, limited detailed guidance, and the lack of evaluation frameworks.",
  main_themes: [
    "Policy variation and lack of standardized MAW guidance across local authorities",
    "Insufficient detail on MAW roles, responsibilities, and staff training",
    "Absence of implementation, monitoring, and outcome evaluation mechanisms for MAW in SUDI prevention",
  ],
  main_findings: [
    "From 152 LAs, 26 unique MAW-related SUDI documents were identified; 19 mentioned MAW but only 4 discussed MAW in detail, and just 15% included a detailed implementation plan, while 73% acknowledged MAW importance but lacked specifics.",
    "MAW staff composition commonly includes midwives and a mix of non-health professionals; training for MAW staff is rare and there is no unified model or formal evaluation framework for MAW-based SUDI prevention.",
  ],
  key_gaps: [
    { description: "No clear, standardized national model for MAW in SUDI prevention; few policies provide detailed implementation plans.", status: "explicit" },
    { description: "Training for MAW staff is inconsistent or missing; no standardized, scalable training framework exists.", status: "explicit" },
    { description: "Lack of rigorous evaluation of MAW implementation and outcomes; no robust evidence on feasibility, fidelity, or impact.", status: "explicit" },
    { description: "No systematic approach to defining 'vulnerable families'; criteria vary widely across local authorities.", status: "implicit" },
    { description: "Operational systems underpinning MAW — information sharing, referral pathways, record-keeping — have not been developed.", status: "implicit" },
  ],
};

const ENTITIES = [
  { entity: "multiagency working (MAW)", type: "concept", freq: "high" },
  { entity: "SUDI prevention", type: "intervention", freq: "high" },
  { entity: "local authorities", type: "institution", freq: "high" },
  { entity: "policy variation", type: "concept", freq: "high" },
  { entity: "staff training", type: "concept", freq: "medium" },
  { entity: "evaluation framework", type: "concept", freq: "medium" },
  { entity: "vulnerable families", type: "population", freq: "medium" },
  { entity: "safe sleep", type: "intervention", freq: "medium" },
  { entity: "safeguarding partnerships", type: "institution", freq: "medium" },
  { entity: "implementation plan", type: "concept", freq: "medium" },
  { entity: "standardized guidance", type: "concept", freq: "low" },
  { entity: "socioeconomic deprivation", type: "concept", freq: "low" },
  { entity: "health professionals", type: "role", freq: "low" },
  { entity: "non-health professionals", type: "role", freq: "low" },
  { entity: "midwives", type: "role", freq: "medium" },
];

const RANKED = [
  { rank:1, id:"claude_fd_1", agent:"claude", dir:"Develop and pilot a standardized national MAW framework for SUDI prevention through a Delphi consensus process involving local authorities, safeguarding partnerships, health professionals, and non-health professionals across regions with varying deprivation levels.", score:0.899, confidence:0.95, entities:8, tier:"primary", reasoning:"The review found only 4 of 26 documents included detailed implementation plans, and no clear model of MAW for SUDI prevention emerged. A Delphi process would systematically aggregate expert knowledge across the fragmented landscape to build consensus on roles, responsibilities, referral pathways, and implementation procedures." },
  { rank:2, id:"openai_fd_1", agent:"openai", dir:"Create a national MAW SUDI prevention coordination body under the Child Safeguarding Practice Review Panel to develop standardized guidance, disseminate best practices from high-performing local authorities, and monitor adoption across England.", score:0.866, confidence:0.92, entities:6, tier:"primary", reasoning:"The 2020 CSPR Panel report recommended MAW but no coordinating mechanism exists. A centralized body could bridge the gap between national recommendation and local implementation by providing templates, benchmarking, and accountability." },
  { rank:3, id:"claude_fd_2", agent:"claude", dir:"Design and evaluate a tiered, role-specific training curriculum for MAW staff in SUDI prevention, incorporating safe sleep knowledge, motivational interviewing with vulnerable families, and cross-agency referral protocols.", score:0.852, confidence:0.93, entities:7, tier:"primary", reasoning:"Only 2 of 26 documents mentioned training provision. The paper identifies 31 discrete job roles under the MAW umbrella. A one-size-fits-all approach would be inadequate — role-specific modules addressing the distinct family touchpoints of each professional group are needed." },
  { rank:4, id:"claude_fd_3", agent:"claude", dir:"Conduct a mixed-methods implementation science study evaluating MAW feasibility, fidelity, and proximate outcomes in local authorities with existing MAW policies versus those without.", score:0.835, confidence:0.94, entities:6, tier:"primary", reasoning:"None of the 26 documents reviewed described any evaluation of MAW implementation or outcomes. A comparative implementation science approach would generate the first evidence on whether MAW approaches actually reach vulnerable families and change behavior." },
  { rank:5, id:"openai_fd_4", agent:"openai", dir:"Extract and codify the implementation models from the four detailed MAW policies (including Hampshire, IoW, Portsmouth and Southampton SCPB) as replicable case studies, and pilot their adoption in LAs that currently lack MAW guidance.", score:0.811, confidence:0.91, entities:6, tier:"primary", reasoning:"Rather than building from scratch, the field can learn from existing exemplars. A structured case study extraction and replication trial would test whether detailed guidance transfers effectively to different local authority settings." },
  { rank:6, id:"openai_fd_7", agent:"openai", dir:"Evaluate the scalability and impact of existing local SUDI prevention campaigns that embed MAW (such as Rochdale's 'Keep Baby Safe' and HIPS 'Every Sleep Counts') through process evaluation and before-after SUDI rate comparisons.", score:0.788, confidence:0.88, entities:6, tier:"primary", reasoning:"Specific local campaigns have already integrated MAW into broader safe sleep initiatives. These represent natural experiments that could be evaluated without requiring new interventions to be designed." },
  { rank:7, id:"claude_fd_5", agent:"claude", dir:"Investigate the integration requirements for information-sharing systems, referral pathways, and record-keeping processes across health and non-health agencies to enable coordinated MAW SUDI prevention.", score:0.773, confidence:0.90, entities:6, tier:"primary", reasoning:"The Discussion explicitly states that systems for identifying issues and referral pathways need modifying, yet none of the reviewed documents described these underpinning systems." },
  { rank:8, id:"openai_fd_2", agent:"openai", dir:"Develop a shared digital platform for cross-agency case management, risk flagging, and safe sleep intervention tracking that enables real-time coordination among MAW professionals working with the same vulnerable family.", score:0.760, confidence:0.89, entities:6, tier:"primary", reasoning:"No referral pathways, no record-keeping processes, no information-sharing protocols exist. A housing officer identifying an unsafe sleep environment needs a mechanism to alert the health visitor and social worker." },
  { rank:9, id:"claude_fd_4", agent:"claude", dir:"Establish a consensus-based, equity-sensitive definition of 'family vulnerability' for SUDI prevention that balances targeting precision with avoidance of stigmatization, informed by family and community voices.", score:0.742, confidence:0.88, entities:5, tier:"primary", reasoning:"Documents showed wide variation in which families were targeted. An equity-sensitive framework with family input would address both targeting inconsistency and the ethical tension around surveillance of vulnerable populations." },
  { rank:10, id:"openai_fd_5", agent:"openai", dir:"Design a participatory action research study co-created with families identified as 'vulnerable' to understand their experiences of MAW engagement, perceived barriers to safe sleep, and preferences for cross-agency support delivery.", score:0.729, confidence:0.90, entities:5, tier:"primary", reasoning:"The paper focuses entirely on the professional perspective — no family voices are included. Understanding how families experience MAW involvement is essential for designing approaches they will engage with rather than avoid." },
  { rank:11, id:"claude_fd_7", agent:"claude", dir:"Examine the knowledge, attitudes, confidence, and self-efficacy of non-health MAW professionals (social workers, housing officers, probation staff, police) regarding SUDI risk factors and safe sleep messaging.", score:0.710, confidence:0.87, entities:5, tier:"primary", reasoning:"The paper identifies MAW increasingly involves non-health professionals who 'may find this work challenging and beyond their current skills.' A baseline assessment would ensure training is appropriately calibrated." },
  { rank:12, id:"claude_fd_6", agent:"claude", dir:"Conduct a longitudinal cohort study tracking SUDI rates in local authorities that adopt standardized MAW prevention versus those using traditional health-professional-only models, with stratification by deprivation index.", score:0.694, confidence:0.85, entities:7, tier:"exploratory", reasoning:"Requires multi-year commitment and sufficient MAW adoption to enable comparison groups. High entity overlap but lower near-term feasibility." },
  { rank:13, id:"openai_fd_3", agent:"openai", dir:"Conduct a cost-effectiveness analysis of MAW SUDI prevention versus traditional health-professional-only approaches to build the economic case for resource allocation.", score:0.680, confidence:0.86, entities:6, tier:"exploratory", reasoning:"Valuable but dependent on having implementation data first; more relevant once pilot evaluations produce cost and outcome data." },
  { rank:14, id:"openai_fd_6", agent:"openai", dir:"Perform a cross-national comparative analysis of multiagency infant death prevention models in countries with similar healthcare structures (Scotland, Wales, Netherlands, Scandinavia).", score:0.651, confidence:0.82, entities:4, tier:"exploratory", reasoning:"Lower entity overlap as the paper focuses exclusively on England. However, the complete absence of evaluation frameworks suggests looking abroad could be highly productive." },
];

const CONNECTED = [
  { rank:1, title:"Safe sleep interventions for reducing SUDI in high-risk populations: a systematic review", type:"Methodological", hue:260 },
  { rank:2, title:"Multiagency safeguarding hubs: implementation and impact on child protection outcomes in England", type:"Structural Parallel", hue:280 },
  { rank:3, title:"Implementation science frameworks for scaling public health interventions across local authorities", type:"Methodological", hue:260 },
  { rank:4, title:"Socioeconomic inequalities in infant mortality: the role of cross-sector partnerships", type:"Thematic", hue:320 },
  { rank:5, title:"The Child Safeguarding Practice Review Panel 2020 report: policy impact assessment", type:"Citation Network", hue:35 },
];

const PIPELINE_STEPS = ["Input", "Summary", "Agents", "Ranking", "Results"];

const css = `
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

* { margin:0; padding:0; box-sizing:border-box; }

:root {
  --bg: #E9E2EE;
  --bg2: #DED6E4;
  --glass: rgba(255,255,255,0.18);
  --glass-hover: rgba(255,255,255,0.28);
  --glass-border: rgba(255,255,255,0.45);
  --glass-strong: rgba(255,255,255,0.35);
  --gold: #C9A84C;
  --gold-light: #E8D5A0;
  --gold-dark: #A68932;
  --mauve: #9B8AAD;
  --mauve-deep: #7B6990;
  --plum: #6B5B7B;
  --text: #3A2E47;
  --text-secondary: #7B6990;
  --text-muted: #A698B5;
  --rose: #C47A7A;
  --sage: #7BA68E;
  --radius: 24px;
  --radius-sm: 16px;
  --serif: 'DM Serif Display', Georgia, serif;
  --sans: 'Inter', -apple-system, sans-serif;
  --mono: 'JetBrains Mono', monospace;
}

body {
  font-family: var(--sans);
  background: var(--bg);
  color: var(--text);
  overflow-x: hidden;
  min-height: 100vh;
}

.app { position: relative; min-height: 100vh; }

/* ─── FLOATING SPHERES ─── */
.sphere {
  position: fixed;
  border-radius: 50%;
  pointer-events: none;
  z-index: 0;
}
.sphere-1 {
  width: 200px; height: 200px;
  top: -70px; left: -90px;
  background: radial-gradient(circle at 35% 35%, #F2D8A0, #E4B96A 40%, #C9963A 70%, #A67620 100%);
  box-shadow: inset -20px -20px 40px rgba(0,0,0,0.12), 8px 12px 40px rgba(201,168,76,0.25);
  animation: floatA 18s ease-in-out infinite;
}
.sphere-2 {
  width: 220px; height: 220px;
  bottom: 60px; right: -40px;
  background: radial-gradient(circle at 40% 30%, #F5E0B8, #E8C88A 40%, #D4A847 70%, #B8892A 100%);
  box-shadow: inset -15px -15px 30px rgba(0,0,0,0.1), 6px 10px 30px rgba(201,168,76,0.2);
  animation: floatB 22s ease-in-out infinite;
}
.sphere-3 {
  width: 150px; height: 150px;
  top: 35%; left: 65%;
  background: radial-gradient(circle at 35% 35%, #F0D8C0, #E2BA8A 45%, #D0A060 80%);
  box-shadow: inset -12px -12px 24px rgba(0,0,0,0.08), 4px 8px 24px rgba(201,168,76,0.15);
  animation: floatC 25s ease-in-out infinite;
  opacity: 0.7;
}
.sphere-4 {
  width: 60px; height: 60px;
  top: 18%; right: 15%;
  background: radial-gradient(circle at 35% 35%, #fff, #F0E8D8 50%, #E0D0B0 100%);
  box-shadow: inset -6px -6px 12px rgba(0,0,0,0.05), 2px 4px 12px rgba(0,0,0,0.08);
  animation: floatD 15s ease-in-out infinite;
}
.bubble {
  position: fixed; border-radius: 50%; pointer-events: none; z-index: 0;
  background: radial-gradient(circle at 30% 30%, rgba(255,255,255,0.7), rgba(255,255,255,0.1));
  border: 1px solid rgba(255,255,255,0.5);
}
.bubble-1 { width:28px; height:28px; top:22%; left:12%; animation: floatD 12s ease-in-out infinite; }
.bubble-2 { width:18px; height:18px; top:55%; right:18%; animation: floatA 16s ease-in-out infinite; opacity:0.6; }
.bubble-3 { width:22px; height:22px; bottom:25%; left:30%; animation: floatC 14s ease-in-out infinite; opacity:0.5; }

@keyframes floatA {
  0%,100% { transform: translate(0,0) rotate(0deg); }
  33% { transform: translate(20px,-30px) rotate(5deg); }
  66% { transform: translate(-15px,20px) rotate(-3deg); }
}
@keyframes floatB {
  0%,100% { transform: translate(0,0) rotate(0deg); }
  33% { transform: translate(-25px,15px) rotate(-4deg); }
  66% { transform: translate(18px,-22px) rotate(3deg); }
}
@keyframes floatC {
  0%,100% { transform: translate(0,0); }
  50% { transform: translate(15px,-18px); }
}
@keyframes floatD {
  0%,100% { transform: translateY(0); }
  50% { transform: translateY(-12px); }
}

/* ─── CONTENT ─── */
.content { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 24px 20px; }

/* ─── GLASS ─── */
.glass {
  background: var(--glass);
  backdrop-filter: blur(40px);
  -webkit-backdrop-filter: blur(40px);
  border: 1px solid var(--glass-border);
  border-radius: var(--radius);
  padding: 32px;
  transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
}
.glass:hover {
  background: var(--glass-hover);
  box-shadow: 0 8px 48px rgba(155,138,173,0.12);
}

/* ─── HERO ─── */
.hero { text-align: center; padding: 70px 20px 40px; animation: riseIn 1s ease-out; }
.hero-eyebrow {
  font-family: var(--mono); font-size: 11px; font-weight: 500;
  letter-spacing: 3px; text-transform: uppercase;
  color: var(--gold-dark); margin-bottom: 16px;
}
.hero h1 {
  font-family: var(--serif); font-size: 64px; font-weight: 400;
  letter-spacing: -1px; line-height: 1; color: var(--text);
}
.hero h1 em {
  font-style: italic; color: var(--gold-dark);
}
.hero-sub {
  font-family: var(--serif); font-style: italic;
  font-size: 16px; color: var(--text-secondary); margin-top: 12px;
}
.hero-desc {
  font-size: 14px; color: var(--text-muted); margin-top: 20px;
  max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.7;
  font-weight: 400;
}

/* ─── INPUT ─── */
.input-section { margin-top: 8px; animation: riseIn 1s ease-out 0.15s both; }
.input-label {
  font-family: var(--mono); font-size: 10px; font-weight: 500;
  letter-spacing: 2.5px; text-transform: uppercase;
  color: var(--mauve-deep); margin-bottom: 14px;
}
.input-row { display: flex; gap: 10px; align-items: center; }
.input-field {
  flex: 1; padding: 16px 22px; border-radius: 16px;
  border: 1.5px solid rgba(155,138,173,0.2);
  background: rgba(255,255,255,0.25);
  backdrop-filter: blur(20px);
  font-size: 15px; font-family: var(--sans); font-weight: 500;
  color: var(--text); outline: none;
  transition: all 0.3s;
}
.input-field:focus { border-color: var(--gold); background: rgba(255,255,255,0.35); }
.input-field::placeholder { color: var(--text-muted); }
.btn-gold {
  padding: 16px 36px; border: none; border-radius: 16px;
  background: linear-gradient(135deg, var(--gold), var(--gold-dark));
  color: #fff; font-size: 14px; font-weight: 700; font-family: var(--sans);
  cursor: pointer; white-space: nowrap; letter-spacing: 0.5px;
  transition: all 0.3s;
  box-shadow: 0 4px 20px rgba(201,168,76,0.3);
}
.btn-gold:hover { transform: translateY(-2px); box-shadow: 0 8px 32px rgba(201,168,76,0.4); }
.btn-gold:disabled { opacity: 0.7; cursor: default; transform: none; }

.paper-info {
  margin-top: 18px; padding: 18px 22px; border-radius: var(--radius-sm);
  background: rgba(201,168,76,0.06);
  border: 1px solid rgba(201,168,76,0.12);
}
.paper-pmid { font-family: var(--mono); font-size: 12px; color: var(--gold-dark); font-weight: 500; }
.paper-title-d { font-family: var(--serif); font-size: 16px; color: var(--text); margin-top: 4px; line-height: 1.4; }
.paper-authors-d { font-size: 12px; color: var(--text-muted); margin-top: 4px; line-height: 1.4; }

/* ─── PIPELINE ─── */
.pipeline {
  display: flex; align-items: center; justify-content: center;
  gap: 0; margin: 36px 0 12px;
  animation: riseIn 0.8s ease-out 0.3s both;
}
.pipe-wrap { position: relative; display: flex; flex-direction: column; align-items: center; }
.pipe-node {
  width: 42px; height: 42px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 13px; font-weight: 700; cursor: pointer;
  border: 2px solid rgba(155,138,173,0.2);
  background: rgba(255,255,255,0.25);
  backdrop-filter: blur(16px);
  color: var(--text-muted);
  transition: all 0.5s cubic-bezier(0.4,0,0.2,1);
}
.pipe-node.active {
  background: linear-gradient(135deg, var(--gold), var(--gold-dark));
  color: #fff; border-color: transparent;
  box-shadow: 0 4px 20px rgba(201,168,76,0.35);
  animation: pulse 1.5s ease-in-out infinite;
}
.pipe-node.done {
  background: var(--sage); color: #fff; border-color: transparent;
  box-shadow: 0 2px 12px rgba(123,166,142,0.3);
}
.pipe-lbl {
  font-size: 10px; font-weight: 600; color: var(--text-muted);
  margin-top: 6px; text-align: center; position: absolute;
  bottom: -22px; left: 50%; transform: translateX(-50%); white-space: nowrap;
  letter-spacing: 0.5px;
}
.pipe-connector {
  width: 50px; height: 2px;
  background: rgba(155,138,173,0.15);
  transition: background 0.5s;
}
.pipe-connector.done {
  background: linear-gradient(90deg, var(--sage), var(--gold));
}

@keyframes pulse {
  0%,100% { box-shadow: 0 4px 20px rgba(201,168,76,0.35); }
  50% { box-shadow: 0 4px 32px rgba(201,168,76,0.55); }
}

/* ─── TABS ─── */
.tabs {
  display: flex; gap: 3px; margin: 44px 0 20px;
  background: rgba(255,255,255,0.12);
  backdrop-filter: blur(24px);
  border-radius: 20px; padding: 5px;
  border: 1px solid var(--glass-border);
}
.tab {
  flex: 1; padding: 13px 6px; border: none; border-radius: 16px;
  background: transparent; font-size: 12.5px; font-weight: 600;
  color: var(--text-muted); cursor: pointer; font-family: var(--sans);
  transition: all 0.4s; letter-spacing: 0.2px;
}
.tab:hover { color: var(--text-secondary); }
.tab.active {
  background: rgba(255,255,255,0.45);
  color: var(--text);
  box-shadow: 0 2px 16px rgba(155,138,173,0.12);
  backdrop-filter: blur(20px);
}

/* ─── ANIMATIONS ─── */
@keyframes riseIn {
  from { opacity: 0; transform: translateY(28px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes slidePanel {
  from { opacity: 0; transform: translateY(14px) scale(0.995); }
  to { opacity: 1; transform: translateY(0) scale(1); }
}
.panel { animation: slidePanel 0.5s cubic-bezier(0.4,0,0.2,1); }

/* ─── SECTION TYPOGRAPHY ─── */
.sec-title { font-family: var(--serif); font-size: 26px; color: var(--text); margin-bottom: 4px; }
.sec-sub { font-size: 13px; color: var(--text-muted); margin-bottom: 24px; line-height: 1.5; }

.tag {
  display: inline-block; font-size: 9px; font-weight: 700;
  text-transform: uppercase; letter-spacing: 1.8px;
  padding: 5px 12px; border-radius: 8px; margin-bottom: 12px;
}
.tag-theme { background: rgba(155,138,173,0.12); color: var(--mauve-deep); }
.tag-finding { background: rgba(123,166,142,0.12); color: var(--sage); }
.tag-gap { background: rgba(196,122,122,0.1); color: var(--rose); }
.tag-implicit { background: rgba(201,168,76,0.1); color: var(--gold-dark); }

.item {
  padding: 16px 20px; border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.18);
  border-left: 3px solid transparent;
  margin-bottom: 8px;
  font-size: 14px; line-height: 1.65; color: #4A3E5A;
  transition: all 0.3s;
}
.item:hover { background: rgba(255,255,255,0.28); }
.item-theme { border-left-color: var(--mauve); }
.item-finding { border-left-color: var(--sage); }
.item-gap-e { border-left-color: var(--rose); }
.item-gap-i { border-left-color: var(--gold); }

.gap-label {
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.2px; margin-right: 8px;
  padding: 2px 8px; border-radius: 4px; display: inline-block;
}
.gap-label-e { background: rgba(196,122,122,0.1); color: var(--rose); }
.gap-label-i { background: rgba(201,168,76,0.1); color: var(--gold-dark); }

/* ─── ENTITIES ─── */
.ent-cloud { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.ent-pill {
  padding: 7px 16px; border-radius: 50px; font-size: 12px; font-weight: 600;
  transition: all 0.3s; cursor: default;
  border: 1.5px solid transparent;
}
.ent-pill:hover { transform: translateY(-2px); }
.ent-high { background: rgba(155,138,173,0.14); color: var(--plum); border-color: rgba(155,138,173,0.25); }
.ent-medium { background: rgba(201,168,76,0.1); color: var(--gold-dark); border-color: rgba(201,168,76,0.2); }
.ent-low { background: rgba(168,160,180,0.1); color: var(--text-muted); border-color: rgba(168,160,180,0.15); }

/* ─── AGENTS ─── */
.agents-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
@media(max-width:768px) { .agents-grid { grid-template-columns: 1fr; } }
.agent-hdr { display: flex; align-items: center; gap: 12px; margin-bottom: 18px; }
.agent-ava {
  width: 40px; height: 40px; border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--serif); font-size: 18px; color: #fff;
}
.ava-claude { background: linear-gradient(135deg, #C9A84C, #A68932); }
.ava-openai { background: linear-gradient(135deg, #7BA68E, #5A8A6E); }
.agent-nm { font-family: var(--serif); font-size: 18px; color: var(--text); }
.agent-mdl { font-family: var(--mono); font-size: 11px; color: var(--text-muted); }
.r-step {
  padding: 14px 18px; border-radius: 14px; margin-bottom: 8px;
  background: rgba(255,255,255,0.18);
  font-size: 13px; line-height: 1.65; color: #5A4E6A;
}
.r-label {
  font-family: var(--mono); font-size: 9px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 1.5px;
  color: var(--mauve-deep); margin-bottom: 6px;
}

/* ─── DIRECTIONS ─── */
.dir-card {
  display: flex; gap: 16px; align-items: flex-start;
  padding: 22px; border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.18);
  backdrop-filter: blur(24px);
  border: 1px solid rgba(255,255,255,0.35);
  margin-bottom: 10px;
  transition: all 0.4s cubic-bezier(0.4,0,0.2,1);
}
.dir-card:hover {
  background: rgba(255,255,255,0.32);
  transform: translateX(4px);
  box-shadow: 0 4px 28px rgba(155,138,173,0.1);
}
.dir-rank-badge {
  min-width: 44px; height: 44px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--serif); font-size: 20px; color: #fff; flex-shrink: 0;
}
.dir-rank-primary { background: linear-gradient(135deg, var(--mauve), var(--plum)); }
.dir-rank-exploratory { background: linear-gradient(135deg, var(--gold), var(--gold-dark)); }
.dir-body { flex: 1; min-width: 0; }
.dir-text { font-size: 14px; line-height: 1.65; color: #4A3E5A; margin-bottom: 12px; }
.dir-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.dir-tag {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 12px; border-radius: 10px;
  font-size: 11px; font-weight: 600;
}
.dt-claude { background: rgba(201,168,76,0.1); color: var(--gold-dark); }
.dt-openai { background: rgba(123,166,142,0.1); color: #5A8A6E; }
.dt-score { background: rgba(155,138,173,0.1); color: var(--mauve-deep); }
.dt-ent { background: rgba(107,91,123,0.08); color: var(--plum); }

.score-track { flex: 1; min-width: 100px; max-width: 180px; height: 5px; border-radius: 3px; background: rgba(155,138,173,0.1); overflow: hidden; }
.score-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--mauve), var(--gold));
  transition: width 1.2s cubic-bezier(0.4,0,0.2,1);
}

.toggle-reason {
  margin-top: 10px; padding: 0; border: none; background: none;
  font-size: 12px; font-weight: 600; color: var(--mauve-deep);
  cursor: pointer; font-family: var(--sans); letter-spacing: 0.3px;
  transition: color 0.2s;
}
.toggle-reason:hover { color: var(--gold-dark); }
.reason-box {
  margin-top: 10px; padding: 14px 18px; border-radius: 12px;
  background: rgba(155,138,173,0.06);
  border-left: 3px solid var(--mauve);
  font-size: 13px; line-height: 1.65; color: #5A4E6A;
  animation: slidePanel 0.35s ease-out;
}

.tier-divider {
  display: flex; align-items: center; gap: 14px; margin: 30px 0 18px;
}
.tier-ln { flex: 1; height: 1px; background: rgba(201,168,76,0.25); }
.tier-tag {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  letter-spacing: 2px; text-transform: uppercase;
  color: var(--gold-dark); padding: 5px 16px;
  background: rgba(201,168,76,0.08); border-radius: 20px;
}

/* ─── CONNECTED ─── */
.conn-item {
  display: flex; align-items: center; gap: 14px;
  padding: 18px 22px; border-radius: var(--radius-sm);
  background: rgba(255,255,255,0.18);
  border-left: 4px solid var(--mauve);
  margin-bottom: 10px;
  transition: all 0.3s;
}
.conn-item:hover { background: rgba(255,255,255,0.3); transform: translateX(4px); }
.conn-num {
  width: 30px; height: 30px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-family: var(--serif); font-size: 14px; color: #fff;
  background: linear-gradient(135deg, var(--mauve), var(--plum));
  flex-shrink: 0;
}
.conn-ttl { font-size: 14px; font-weight: 600; color: var(--text); line-height: 1.4; }
.conn-tp {
  display: inline-block; margin-top: 5px;
  font-size: 9px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 1.5px; padding: 3px 10px; border-radius: 6px; color: #fff;
}

/* ─── MRR ─── */
.mrr-block {
  text-align: center; padding: 28px;
  background: rgba(255,255,255,0.12); border-radius: var(--radius-sm);
  margin-top: 20px;
}
.mrr-val {
  font-family: var(--serif); font-size: 56px; color: var(--text);
  line-height: 1;
}
.mrr-lbl {
  font-family: var(--mono); font-size: 10px; letter-spacing: 2px;
  text-transform: uppercase; color: var(--text-muted); margin-top: 6px;
}
.mrr-eq {
  font-family: var(--mono); font-size: 12px; color: var(--text-muted);
  margin-top: 14px; padding: 8px 20px; background: rgba(255,255,255,0.2);
  border-radius: 10px; display: inline-block;
}

/* ─── LOADING ─── */
.loading-area { text-align: center; padding: 70px 20px; }
.ld-ring {
  width: 52px; height: 52px; margin: 0 auto 18px;
  border: 3px solid rgba(201,168,76,0.15);
  border-top-color: var(--gold);
  border-radius: 50%;
  animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.ld-text { font-size: 14px; font-weight: 500; color: var(--text-muted); font-style: italic; }

/* ─── INFO BOX ─── */
.info-box {
  margin-top: 24px; padding: 18px 22px; border-radius: 14px;
  background: rgba(255,255,255,0.12);
  border: 1px solid rgba(255,255,255,0.2);
}
.info-title {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  letter-spacing: 2px; text-transform: uppercase;
  color: var(--mauve-deep); margin-bottom: 8px;
}
.info-body { font-size: 13px; line-height: 1.7; color: var(--text-secondary); }

.legend { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 20px; }
.legend-item { display: flex; align-items: center; gap: 6px; }
`;

export default function Foresight() {
  const [stage, setStage] = useState("idle");
  const [activeTab, setActiveTab] = useState(0);
  const [pipeStep, setPipeStep] = useState(0);
  const [inputVal, setInputVal] = useState("11812686");
  const [expandedDir, setExpandedDir] = useState(null);
  const tabs = ["Summary", "Entities", "Agents", "Directions", "Connected"];

  const run = () => {
    setStage("loading"); setPipeStep(0);
    let s = 0;
    const iv = setInterval(() => {
      s++;
      setPipeStep(s);
      if (s >= 5) { clearInterval(iv); setTimeout(() => { setStage("done"); setActiveTab(0); }, 600); }
    }, 750);
  };

  return (
    <>
      <style>{css}</style>
      <div className="app">
        <div className="sphere sphere-1" />
        <div className="sphere sphere-2" />
        <div className="sphere sphere-3" />
        <div className="sphere sphere-4" />
        <div className="bubble bubble-1" />
        <div className="bubble bubble-2" />
        <div className="bubble bubble-3" />

        <div className="content">
          <div className="hero">
            <div className="hero-eyebrow">Multi-Agent Research System</div>
            <h1><em>Foresight</em></h1>
            <div className="hero-sub">
              Future-Oriented Research Engine for Scientific Insight, Guidance & Hierarchical Targeting
            </div>
            <p className="hero-desc">
              An agentic system that reads research papers, reasons through gaps via dual AI agents,
              and surfaces ranked future directions — powered by entity-driven rank fusion.
            </p>
          </div>

          <div className="glass input-section">
            <div className="input-label">Paper Input</div>
            <div className="input-row">
              <input className="input-field" value={inputVal} onChange={e=>setInputVal(e.target.value)} placeholder="Enter PubMed ID…" disabled={stage==="loading"} />
              <button className="btn-gold" onClick={run} disabled={stage==="loading"}>
                {stage==="loading" ? "Analyzing…" : "Run Foresight"}
              </button>
            </div>
            {stage !== "idle" && (
              <div className="paper-info">
                <div className="paper-pmid">PubMed {PAPER_DATA.id}</div>
                <div className="paper-title-d">{PAPER_DATA.title}</div>
                <div className="paper-authors-d">{PAPER_DATA.authors}</div>
              </div>
            )}
          </div>

          {stage !== "idle" && (
            <div className="pipeline">
              {PIPELINE_STEPS.map((s,i) => (
                <div style={{display:"flex",alignItems:"center"}} key={i}>
                  <div className="pipe-wrap">
                    <div className={`pipe-node ${pipeStep>i?"done":pipeStep===i&&stage==="loading"?"active":""}`}>
                      {pipeStep>i ? "✓" : i+1}
                    </div>
                    <div className="pipe-lbl">{s}</div>
                  </div>
                  {i < PIPELINE_STEPS.length-1 && <div className={`pipe-connector ${pipeStep>i?"done":""}`} />}
                </div>
              ))}
            </div>
          )}

          {stage === "loading" && (
            <div className="loading-area">
              <div className="ld-ring" />
              <div className="ld-text">
                {pipeStep===0 && "Fetching paper from PubMed…"}
                {pipeStep===1 && "Agent 1 generating structured summary…"}
                {pipeStep===2 && "Claude & OpenAI agents reasoning…"}
                {pipeStep===3 && "Running entity-driven rank fusion…"}
                {pipeStep===4 && "Enriching with Connected Papers…"}
              </div>
            </div>
          )}

          {stage === "done" && (
            <>
              <div className="tabs">
                {tabs.map((t,i) => (
                  <button key={i} className={`tab ${activeTab===i?"active":""}`} onClick={()=>setActiveTab(i)}>{t}</button>
                ))}
              </div>
              <div className="panel" key={activeTab}>
                {activeTab===0 && <SummaryTab />}
                {activeTab===1 && <EntitiesTab />}
                {activeTab===2 && <AgentsTab />}
                {activeTab===3 && <DirectionsTab exp={expandedDir} setExp={setExpandedDir} />}
                {activeTab===4 && <ConnectedTab />}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}

/* ─── TAB PANELS ─── */

function SummaryTab() {
  return (
    <div className="glass">
      <div className="sec-title">Structured Summary</div>
      <div className="sec-sub">Generated by the Summarization Agent from the source paper</div>

      <div style={{marginBottom:24}}>
        <div className="tag tag-theme">Topic & Scope</div>
        <p style={{fontSize:14,lineHeight:1.7,color:"#5A4E6A"}}>{SUMMARY.topic_and_scope}</p>
      </div>

      <div style={{marginBottom:24}}>
        <div className="tag tag-theme">Main Themes</div>
        {SUMMARY.main_themes.map((t,i) => <div className="item item-theme" key={i}>{t}</div>)}
      </div>

      <div style={{marginBottom:24}}>
        <div className="tag tag-finding">Main Findings</div>
        {SUMMARY.main_findings.map((f,i) => <div className="item item-finding" key={i}>{f}</div>)}
      </div>

      <div>
        <div className="tag tag-gap">Key Gaps</div>
        {SUMMARY.key_gaps.map((g,i) => (
          <div className={`item ${g.status==="explicit"?"item-gap-e":"item-gap-i"}`} key={i}>
            <span className={`gap-label ${g.status==="explicit"?"gap-label-e":"gap-label-i"}`}>{g.status}</span>
            {g.description}
          </div>
        ))}
      </div>
    </div>
  );
}

function EntitiesTab() {
  return (
    <div className="glass">
      <div className="sec-title">Extracted Entities</div>
      <div className="sec-sub">Relevance signals drawn from the title, abstract, and structured summary</div>

      <div className="legend">
        {[{l:"High frequency",c:"ent-high"},{l:"Medium",c:"ent-medium"},{l:"Low",c:"ent-low"}].map(x=>(
          <div className="legend-item" key={x.c}><span className={`ent-pill ${x.c}`} style={{padding:"3px 10px",fontSize:11}}>{x.l}</span></div>
        ))}
      </div>

      <div className="ent-cloud">
        {ENTITIES.map((e,i)=>(
          <span className={`ent-pill ent-${e.freq}`} key={i} title={e.type}>{e.entity}</span>
        ))}
      </div>

      <div className="info-box">
        <div className="info-title">How entities drive ranking</div>
        <div className="info-body">
          Each candidate future direction is scored by entity overlap with these extracted terms.
          The entity relevance score contributes 60% of the final fused score, with the remaining 40% from the agent's own ranking position. Directions with higher overlap are promoted; those with less overlap are placed in an exploratory tier.
        </div>
      </div>
    </div>
  );
}

function AgentsTab() {
  const data = [
    { name:"Claude", mdl:"claude-sonnet-4-6", cls:"ava-claude", letter:"C", count:7,
      steps:[
        {l:"Gap Analysis", t:"The paper reveals three critical gaps: (1) no standardized national model — only 4 of 26 documents provided detailed implementation plans; (2) training is inconsistent, with only 2 documents mentioning it; (3) no evaluation exists. An implicit gap is the wide variation in 'vulnerable family' definitions."},
        {l:"Theme Synthesis", t:"MAW is increasingly recommended (73% acknowledge its importance), but the operational infrastructure — standardized models, training, evaluation, systems integration — is almost entirely absent. The field is at an inflection point."},
        {l:"Direction Generation", t:"Directions address root causes: evidence-based model development via Delphi consensus, scalable role-specific training, implementation science evaluation, systems-level integration, and equity-sensitive vulnerability definitions."}
      ]},
    { name:"OpenAI", mdl:"gpt-5-nano-2025-08-07", cls:"ava-openai", letter:"O", count:7,
      steps:[
        {l:"Gap Analysis", t:"Three explicit gaps dominate: absence of a unified MAW model despite 73% acknowledgment; training in only 2 of 26 documents; zero evaluation frameworks. An implicit gap is the lack of digital infrastructure for cross-agency coordination."},
        {l:"Theme Synthesis", t:"A classic policy-practice gap: top-down recommendations diffused awareness (policies doubled 2020–2022), but bottom-up capacity hasn't kept pace. The 31 job roles and absent systems planning demand both horizontal coordination and vertical integration."},
        {l:"Direction Generation", t:"Targets structural barriers: national coordination body, shared digital platform, community-centered participatory research, cost-effectiveness analysis, cross-national learning, and evaluating existing campaigns like Rochdale's 'Keep Baby Safe'."}
      ]}
  ];
  return (
    <div className="agents-grid">
      {data.map(a=>(
        <div className="glass" key={a.name}>
          <div className="agent-hdr">
            <div className={`agent-ava ${a.cls}`}>{a.letter}</div>
            <div>
              <div className="agent-nm">{a.name} Agent</div>
              <div className="agent-mdl">{a.mdl}</div>
            </div>
          </div>
          <div style={{fontSize:12,fontWeight:600,color:"var(--text-muted)",marginBottom:8}}>{a.count} directions generated</div>
          {a.steps.map((s,i)=>(
            <div className="r-step" key={i}>
              <div className="r-label">Step {i+1} · {s.l}</div>
              {s.t}
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function DirectionsTab({ exp, setExp }) {
  const primary = RANKED.filter(d=>d.tier==="primary");
  const exploratory = RANKED.filter(d=>d.tier==="exploratory");
  return (
    <div>
      <div className="glass" style={{marginBottom:16}}>
        <div className="sec-title">Fused & Ranked Directions</div>
        <div className="sec-sub">14 directions from 2 agents — ranked by Reciprocal Rank Fusion (60% entity · 40% agent rank)</div>
        <div className="mrr-block">
          <div className="mrr-val">0.899</div>
          <div className="mrr-lbl">Top Fused Score</div>
          <div className="mrr-eq">MRR = (1/|Q|) × Σ(1/rank_i)</div>
        </div>
      </div>

      {primary.map(d=><DirCard key={d.id} d={d} open={exp===d.id} toggle={()=>setExp(exp===d.id?null:d.id)} />)}

      <div className="tier-divider">
        <div className="tier-ln" />
        <div className="tier-tag">Exploratory</div>
        <div className="tier-ln" />
      </div>

      {exploratory.map(d=><DirCard key={d.id} d={d} open={exp===d.id} toggle={()=>setExp(exp===d.id?null:d.id)} />)}
    </div>
  );
}

function DirCard({ d, open, toggle }) {
  return (
    <div className="dir-card">
      <div className={`dir-rank-badge ${d.tier==="primary"?"dir-rank-primary":"dir-rank-exploratory"}`}>{d.rank}</div>
      <div className="dir-body">
        <div className="dir-text">{d.dir}</div>
        <div className="dir-meta">
          <span className={`dir-tag ${d.agent==="claude"?"dt-claude":"dt-openai"}`}>
            {d.agent==="claude"?"◈ Claude":"◉ OpenAI"}
          </span>
          <span className="dir-tag dt-score">{d.score.toFixed(3)}</span>
          <span className="dir-tag dt-ent">{d.entities} entities</span>
          <div className="score-track"><div className="score-fill" style={{width:`${d.score*100}%`}} /></div>
        </div>
        <button className="toggle-reason" onClick={toggle}>{open?"▾ Hide reasoning":"▸ Show reasoning"}</button>
        {open && <div className="reason-box">{d.reasoning}</div>}
      </div>
    </div>
  );
}

function ConnectedTab() {
  const hueToColor = h => `hsl(${h},40%,45%)`;
  return (
    <div className="glass">
      <div className="sec-title">Connected Papers</div>
      <div className="sec-sub">Related papers from the citation graph that enrich direction ranking</div>
      {CONNECTED.map(c=>(
        <div className="conn-item" key={c.rank} style={{borderLeftColor:hueToColor(c.hue)}}>
          <div className="conn-num" style={{background:`linear-gradient(135deg, hsl(${c.hue},40%,50%), hsl(${c.hue},40%,38%))`}}>{c.rank}</div>
          <div style={{flex:1}}>
            <div className="conn-ttl">{c.title}</div>
            <span className="conn-tp" style={{background:hueToColor(c.hue)}}>{c.type}</span>
          </div>
        </div>
      ))}
      <div className="info-box">
        <div className="info-title">Integration with Ranking</div>
        <div className="info-body">
          Directions that align with trends across connected papers receive a ranking boost,
          rewarding suggestions that bridge or extend existing research clusters. This enrichment
          layer surfaces directions that are both novel and grounded in the broader literature.
        </div>
      </div>
    </div>
  );
}
