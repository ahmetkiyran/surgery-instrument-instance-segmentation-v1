from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "V1_Scientific_Manuscript_Revised_2026-09-25.pdf"


class ArchitectureFigure(Flowable):
    def __init__(self):
        super().__init__()
        self.width = 16.2 * cm
        self.height = 4.2 * cm

    def draw(self):
        c = self.canv
        navy = colors.HexColor("#17365D")
        blue = colors.HexColor("#D9EAF7")
        pale = colors.HexColor("#F4F7FA")
        boxes = [
            (0.0, 2.55, 3.1, 0.7, "CLI / Gradio"),
            (0.0, 1.10, 3.1, 0.7, "Tauri / Expo"),
            (5.0, 1.82, 4.2, 0.9, "Unified Python core"),
            (11.0, 2.55, 4.2, 0.7, "Inspection artifacts"),
            (11.0, 1.10, 4.2, 0.7, "Geometry-only privacy artifacts"),
        ]
        for x, y, w, h, label in boxes:
            c.setStrokeColor(navy)
            c.setFillColor(blue if x in (0.0, 5.0) else pale)
            c.roundRect(x * cm, y * cm, w * cm, h * cm, 5, fill=1, stroke=1)
            c.setFillColor(colors.black)
            c.setFont("Helvetica", 8)
            c.drawCentredString((x + w / 2) * cm, (y + 0.27) * cm, label)
        c.setStrokeColor(navy)
        c.setLineWidth(1.1)
        for start, end in [((3.1, 2.9), (5.0, 2.27)), ((3.1, 1.45), (5.0, 2.27)), ((9.2, 2.27), (11.0, 2.9)), ((9.2, 2.27), (11.0, 1.45))]:
            c.line(start[0] * cm, start[1] * cm, end[0] * cm, end[1] * cm)
        c.setFillColor(colors.HexColor("#5E6B75"))
        c.setFont("Helvetica-Oblique", 7)
        c.drawCentredString(13.1 * cm, 0.35 * cm, "The privacy renderer receives geometry and pose metadata, never source RGB.")


def page_header_footer(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#A8B4C0"))
    canvas.line(1.7 * cm, height - 1.35 * cm, width - 1.7 * cm, height - 1.35 * cm)
    canvas.setFillColor(colors.HexColor("#4F5B66"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(1.7 * cm, height - 1.08 * cm, "Tabletop instrument visibility audit | Blinded manuscript")
    canvas.drawRightString(width - 1.7 * cm, height - 1.08 * cm, "Research article")
    canvas.drawCentredString(width / 2, 1.05 * cm, f"Research article  |  {doc.page}")
    canvas.restoreState()


def p(text, style):
    return Paragraph(text, style)


def section(title, body, styles, story):
    story.append(Spacer(1, 0.14 * cm))
    story.append(p(title, styles["H1"]))
    for paragraph in body:
        story.append(p(paragraph, styles["Body"]))


def table(rows, widths, styles, font_size=7.5):
    prepared = [[p(str(cell), styles["TableHead"] if i == 0 else styles["TableCell"]) for cell in row] for i, row in enumerate(rows)]
    value = LongTable(prepared, colWidths=widths, repeatRows=1, hAlign="LEFT")
    value.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C8D0D8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FB")]),
    ]))
    return value


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT), pagesize=A4, rightMargin=1.7 * cm, leftMargin=1.7 * cm,
        topMargin=1.75 * cm, bottomMargin=1.55 * cm,
        title="Software verification of a multi-client surgical video analysis pipeline",
        author="Blinded manuscript",
    )
    base = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=8),
        "Subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10, leading=13, alignment=TA_CENTER, textColor=colors.HexColor("#4F5B66"), spaceAfter=14),
        "H1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=12, leading=15, textColor=colors.HexColor("#17365D"), spaceBefore=6, spaceAfter=5),
        "H2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10, leading=13, textColor=colors.HexColor("#17365D"), spaceBefore=5, spaceAfter=3),
        "Body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, alignment=TA_JUSTIFY, spaceAfter=6),
        "Abstract": ParagraphStyle("Abstract", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, alignment=TA_JUSTIFY, leftIndent=0.25 * cm, rightIndent=0.25 * cm, spaceAfter=5),
        "Caption": ParagraphStyle("Caption", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10.5, alignment=TA_LEFT, spaceBefore=3, spaceAfter=7),
        "TableHead": ParagraphStyle("TableHead", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=7.4, leading=9, textColor=colors.white, alignment=TA_LEFT),
        "TableCell": ParagraphStyle("TableCell", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2, alignment=TA_LEFT),
        "Reference": ParagraphStyle("Reference", parent=base["BodyText"], fontName="Helvetica", fontSize=7.4, leading=9.2, leftIndent=0.45 * cm, firstLineIndent=-0.45 * cm, spaceAfter=2),
        "Note": ParagraphStyle("Note", parent=base["BodyText"], fontName="Helvetica-Oblique", fontSize=8, leading=10.5, spaceAfter=6),
    }
    story = []
    story += [
        Spacer(1, 0.25 * cm),
        p("Software Verification of a Multi-Client Surgical Video Analysis Pipeline", styles["Title"]),
        p("A revised engineering validation with evidence boundaries and literature-informed citations", styles["Subtitle"]),
        p("Blinded manuscript for peer review", styles["Subtitle"]),
        p("Abstract", styles["H1"]),
        p("<b>Background.</b> Surgical-video analytics can support descriptive workflow research, yet software verification does not establish clinical validity, retained-item detection, or legal anonymization.", styles["Abstract"]),
        p("<b>Objective.</b> To document the current implementation and reproducible verification state of a V1 pipeline for tabletop instrument visibility analysis, optional point-prompted video mask propagation, and privacy-oriented geometry rendering across command-line, web, desktop, and mobile clients.", styles["Abstract"]),
        p("<b>Methods.</b> We reviewed source code, model metadata, automated tests, build outputs, and local machine-readable artifacts. We reran the Python test suite, static checks, TypeScript contracts, desktop web build, Android export, and native Tauri Cargo check. The available SAM3 runtime artifact used a 10-s synthetic moving-object fixture; no clinical-video accuracy experiment was rerun or newly claimed. References were updated through targeted searches of primary publications and authoritative records.", styles["Abstract"]),
        p("<b>Results.</b> Python verification yielded 67 passed tests and 1 skipped opt-in release-integration test; the single warning was a Starlette/httpx deprecation. Ruff and pip check passed. API-client, desktop, and mobile TypeScript checks passed. The Vite build and Android Hermes export passed; Vite retained third-party Lottie direct-eval and chunk-size warnings. Cargo check passed after removal of an unused import. The real SAM3 integration artifact processed 20 synthetic frames with two targets and 35 mask records. Browser automation and physical device validation could not run because no browser automation runtime, Android Debug Bridge, emulator, or connected device was available.", styles["Abstract"]),
        p("<b>Conclusion.</b> The evidence supports an auditable software integration proof of concept. Clinical performance, temporal accuracy on clinical video, identity disclosure risk, and deployment safety remain unestablished and require independently governed evaluation.", styles["Abstract"]),
        p("<b>Keywords.</b> surgical data science; instance segmentation; multi-object tracking; promptable video segmentation; software verification; privacy-oriented rendering; reproducibility", styles["Abstract"]),
    ]

    section("1. Introduction", [
        "Surgical data science aims to convert heterogeneous perioperative data into quantitative descriptions of interventions and outcomes [1]. Computer vision can measure the presence, location, and duration of visible instruments, but such measurements are exposed to occlusion, glare, smoke, camera viewpoint changes, domain shift, and incomplete annotations [2-4]. A visual non-detection is therefore not evidence that an item is absent from the operating room, has been handed off, contacted tissue, or has been retained.",
        "This work reports an engineering verification of an existing repository. It addresses whether the current software provides a consistent, auditable route from video frames to visibility intervals and selected-target artifacts across four client surfaces. It does not evaluate a clinical endpoint. The scope is intentionally separate from retained surgical item prevention, which remains a clinical workflow and accounting responsibility [12-14].",
    ], styles, story)

    section("2. Materials and Methods", [
        "We conducted a retrospective repository and artifact review on 25 September 2026. The evidence base comprised the versioned Python, TypeScript, and Rust sources; configuration; model manifest; automated tests; build products; and local JSON artifacts. No model training, weight modification, new clinical-video acquisition, or ground-truth annotation was performed.",
        "The code resolves instrument and personnel class names from model metadata rather than fixed class identifiers. The pose model exposes one person category with 17 COCO keypoints [9]. Segmentation checkpoint scores, where embedded in a checkpoint, are treated as provenance metadata only because no evaluation dataset, split definition, or scoring script was available for independent reproduction.",
    ], styles, story)
    story.append(p("2.1 System architecture", styles["H2"]))
    story.append(ArchitectureFigure())
    story.append(p("Figure 1. The four clients use one Python analysis core. The privacy renderer receives geometry and pose metadata; the source RGB frame is not supplied to that renderer.", styles["Caption"]))
    section("2.2 Analysis and privacy contracts", [
        "The legacy path writes detections, track identities, class-level visibility unions, instance-time sums, and complement intervals. The complement interval is defined only over processed frames in which the specified class was not detected. It does not attribute the absence to a physical or clinical event. BoT-SORT is the default tracker and ByteTrack is available as an alternative [7,8].",
        "The unified path records normalized point selections, validates that a click lies in the displayed source area, and stores the selection before propagation begins. The external SAM3 adapter performs selected-target propagation when enabled. Promptable image and video segmentation is an active research area [16,17]; this implementation reports its software behavior, not a benchmark result for any foundation model.",
        "The PrivacyXRayRenderer accepts no source frame. It receives geometry, pose, labels, and temporal metadata, and creates a synthetic dark canvas with skeletons, instrument geometry, labels, and trails. The resulting image is neither a medical radiograph nor a legal de-identification guarantee. Operating-room video can remain identifying after visual transformation because context and associated artifacts may disclose identity [15].",
    ], styles, story)

    story.append(p("2.3 Verified model metadata", styles["H2"]))
    story.append(table([
        ["Role", "Declared task", "Verified metadata in manifest/runtime"],
        ["Health personnel", "YOLO11l-seg / segment", "health_personel; light; monitor"],
        ["Surgical instruments", "YOLO11s-seg / segment", "skin_stapler; thumb_forceps; ring_instrument; scissors"],
        ["Pose", "YOLO11m-pose / pose", "person; 17 COCO keypoints"],
    ], [3.2 * cm, 4.0 * cm, 9.0 * cm], styles))
    story.append(p("Table 1. Runtime metadata used to validate model role and output schema. The study did not train these models.", styles["Caption"]))

    section("3. Verification Protocol", [
        "The following commands were rerun locally: Python pytest, Ruff, pip check, API-client/desktop/mobile TypeScript checks, desktop Vite build, Expo Android export, and Cargo check for the Tauri manifest. Android export first failed because the sandbox could not execute the Hermes compiler; the same export passed in the elevated Windows run. Cargo check initially identified an unused Rust import, which was removed before the final successful check.",
        "The available real SAM3 artifact was intentionally retained as a synthetic moving-object integration fixture. It comprised 20 frames at 128 x 96 pixels and 2 frames/s, with two selected targets. It is suitable for checking cache merge, artifact production, and privacy flags, but is not clinical-video validation. Earlier manuscript statements about a clinical first-minute run are excluded from this revision because the authorized source video and its reproducible run package are not present in the reviewed repository.",
    ], styles, story)

    story.append(p("3.1 Verification matrix", styles["H2"]))
    story.append(table([
        ["Surface", "Result", "Evidence boundary"],
        ["Python tests", "67 passed; 1 skipped; 1 warning", "Local pytest. Skipped test requires a published release and authorized smoke video."],
        ["Ruff and pip check", "Passed", "Static Python lint and installed dependency consistency."],
        ["API, desktop, mobile typecheck", "Passed", "Shared TypeScript contract compilation."],
        ["Desktop Vite build", "Passed", "Third-party Lottie direct-eval and chunk-size warnings remain."],
        ["Expo Android export", "Passed", "Hermes bytecode produced in an elevated Windows run; no device was launched."],
        ["Native Tauri Cargo check", "Passed", "Rust compilation check completed; no installer or interactive WebView session was tested."],
        ["Browser automation", "Not executed", "No Playwright/browser automation runtime was installed."],
        ["Physical Android/iOS", "Not executed", "ADB, emulator, and connected devices were unavailable."],
        ["GitHub publication", "Not a test", "Commit and push are release operations, reported separately from technical verification."],
    ], [3.6 * cm, 3.0 * cm, 9.6 * cm], styles))
    story.append(p("Table 2. Current verification results and their limits.", styles["Caption"]))

    section("4. Results", [
        "The verification suite completed without a test failure when run with an isolated Windows temporary directory. The warning came from the FastAPI test client stack: Starlette deprecates its current httpx TestClient usage. It does not identify a functional failure in the repository tests. The desktop build emitted two non-blocking warnings associated with the bundled Lottie dependency and a minified JavaScript chunk larger than 500 kB.",
        "The native Tauri check completed successfully after removing the unused Manager import from the desktop entry point. This establishes Rust source compilation for the configured manifest. It does not establish packaged installer behavior, code signing, WebView behavior, or sidecar lifecycle under an interactive desktop session.",
    ], styles, story)
    story.append(p("4.1 Synthetic SAM3 integration artifact", styles["H2"]))
    story.append(table([
        ["Measure", "Observed value", "Interpretation"],
        ["Input", "20 synthetic frames; 128 x 96; 2 frames/s", "Moving-object fixture, not a clinical dataset."],
        ["Target selections", "2", "One person target and one instrument target."],
        ["Successful mask records", "35", "20 person and 15 instrument records; record count is not accuracy."],
        ["Lost or recovery frames", "0 / 0", "Fixture behavior only; cannot estimate robustness to surgical occlusion."],
        ["Privacy flags", "RGB false; copied RGB false; audio false", "Manifest and privacy report record the renderer contract."],
    ], [4.0 * cm, 4.0 * cm, 8.2 * cm], styles))
    story.append(p("Table 3. Machine-readable properties of outputs/sam3_real_e2e. Counts describe one synthetic fixture and must not be interpreted as sensitivity, precision, recall, or clinical accuracy.", styles["Caption"]))

    section("5. Discussion", [
        "The revised evidence supports the narrower claim that the repository's major software surfaces share a testable contract: clients create sessions and selections, the Python core performs analysis, and artifacts preserve configuration and privacy-related flags. This is useful engineering evidence because surgical AI systems can otherwise diverge across interfaces. It does not resolve performance in real operations, where annotation provenance and procedure-level separation are essential [10,11].",
        "The implementation keeps a useful boundary between descriptive visibility and clinical meaning. A not-visible interval can be affected by camera framing, detector threshold, mask propagation, occlusion, or a person covering an instrument. Future clinical studies should establish the initial inventory, annotate departures, returns, occlusions, and hand-offs per instance, maintain patient- and procedure-level separation, and report event sensitivity, precision, onset/offset error, duration error, and class-stratified failures.",
        "Privacy is similarly bounded. The renderer's source-frame prohibition is a verifiable software property, while anonymization is a contextual socio-technical claim. Source-derived inspection views, motion patterns, timestamps, logs, and ancillary files require governance and human review before dissemination [15].",
    ], styles, story)

    section("6. Limitations", [
        "No independent clinical dataset, temporal ground truth, data split, or leakage analysis was available. The SAM3 evidence is a 10-s synthetic fixture and cannot support claims about surgical tool segmentation, long-video continuity, clinical occlusion, or generalization. Browser automation, physical mobile devices, iOS behavior, interactive Tauri execution, installer packaging, and release distribution were not executed in the available host. The Android export validates a bundle, not runtime behavior on a device. The privacy-oriented visualization is not medical imaging and does not establish HIPAA, GDPR, KVKK, or institutional compliance.",
    ], styles, story)

    section("7. Conclusion", [
        "The current V1 repository provides a modular path for tabletop visibility analysis, optional selected-target propagation, and a geometry-only privacy rendering contract across CLI, Gradio, Tauri, and Expo clients. The repeated local verification supports an engineering integration conclusion. Clinical decision support, retained-item inference, deployment safety, full anonymity, and clinical tracking accuracy require independent data, ground-truth labels, prospective evaluation, and governance documentation.",
    ], styles, story)

    section("8. Data and Code Availability", [
        "The code repository is available at https://github.com/ahmetkiyran/surgery-instrument-instance-segmentation-v1. The reviewed repository intentionally excludes source clinical videos, user-specific logs, credentials, audit records, and vendor model environments. The synthetic integration artifact is locally generated and is not a clinical dataset. Reproduction of optional SAM3 behavior requires authorized access to the external runtime, checkpoint, and dependencies.",
    ], styles, story)

    story.append(p("References", styles["H1"]))
    references = [
        "1. Maier-Hein L, Vedula SS, Speidel S, Navab N, Kikinis R, Park A, et al. Surgical data science for next-generation interventions. Nature Biomedical Engineering. 2017;1:691-696. doi:10.1038/s41551-017-0132-7.",
        "2. Twinanda AP, Shehata S, Mutter D, Marescaux J, de Mathelin M, Padoy N. EndoNet: a deep architecture for recognition tasks on laparoscopic videos. IEEE Transactions on Medical Imaging. 2017;36:86-97. doi:10.1109/TMI.2016.2593957.",
        "3. Kennedy-Metz LR, Mascagni P, Torralba A, Dias RD, Perona P, Shah JA, et al. Computer vision in the operating room: opportunities and caveats. IEEE Transactions on Medical Robotics and Bionics. 2021;3:2-10. doi:10.1109/TMRB.2020.3040002.",
        "4. Yang C, Zhao Z, Hu S. Image-based laparoscopic tool detection and tracking using convolutional neural networks: a review of the literature. Computer Assisted Surgery. 2020;25:15-28. doi:10.1080/24699322.2020.1801842.",
        "5. He K, Gkioxari G, Dollar P, Girshick R. Mask R-CNN. Proceedings of ICCV. 2017:2961-2969. doi:10.1109/ICCV.2017.322.",
        "6. Ronneberger O, Fischer P, Brox T. U-Net: convolutional networks for biomedical image segmentation. MICCAI. 2015:234-241. doi:10.1007/978-3-319-24574-4_28.",
        "7. Zhang Y, Sun P, Jiang Y, Yu D, Weng F, Yuan Z, et al. ByteTrack: multi-object tracking by associating every detection box. ECCV. 2022:1-21. doi:10.1007/978-3-031-20047-2_1.",
        "8. Aharon N, Orfaig R, Bobrovsky B-Z. BoT-SORT: robust associations multi-pedestrian tracking. arXiv:2206.14651. 2022. doi:10.48550/arXiv.2206.14651.",
        "9. Lin T-Y, Maire M, Belongie S, Hays J, Perona P, Ramanan D, et al. Microsoft COCO: common objects in context. ECCV. 2014:740-755. doi:10.1007/978-3-319-10602-1_48.",
        "10. Hong W-Y, Kao C-L, Kuo Y-H, Wang J-R, Chang W-L, Shih C-S. CholecSeg8k: a semantic segmentation dataset for laparoscopic cholecystectomy based on Cholec80. arXiv:2012.12453. 2020. doi:10.48550/arXiv.2012.12453.",
        "11. Alabi O, Toe KKZ, Zhou Z, Budd C, Raison N, Shi M, Vercauteren T. CholecInstanceSeg: a tool instance segmentation dataset for laparoscopic surgery. Scientific Data. 2025;12:825. doi:10.1038/s41597-025-05163-w.",
        "12. Sirihorachai R, Saylor KM, Manojlovich M. Interventions for the prevention of retained surgical items: a systematic review. World Journal of Surgery. 2022;46:370-381. doi:10.1007/s00268-021-06370-3.",
        "13. World Health Organization. Implementation manual: WHO surgical safety checklist 2009. Geneva: World Health Organization; 2009. ISBN:9789241598590.",
        "14. The Joint Commission. Sentinel Event Alert 51: preventing unintended retained foreign objects. Oakbrook Terrace, IL: The Joint Commission; 2013.",
        "15. Silas MR, Grassia P, Langerman A. Video recording of the operating room - is anonymity possible? Journal of Surgical Research. 2015;197:272-276. doi:10.1016/j.jss.2015.03.097.",
        "16. Kirillov A, Mintun E, Ravi N, Mao H, Rolland C, Gustafson L, et al. Segment Anything. Proceedings of ICCV. 2023:4015-4026. doi:10.1109/ICCV51070.2023.00371.",
        "17. Ravi N, Gabeur V, Hu Y-T, Hu R, Ryali C, Ma T, et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714. 2024. doi:10.48550/arXiv.2408.00714.",
    ]
    for ref in references:
        story.append(p(ref, styles["Reference"]))
    doc.build(story, onFirstPage=page_header_footer, onLaterPages=page_header_footer)
    print(OUT)


if __name__ == "__main__":
    build()
