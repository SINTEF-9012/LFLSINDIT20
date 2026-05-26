"""
LLM Image Explanation Utilities for CNC Machine Data Visualization.

This module provides utilities for generating comprehensive explanations of
CNC machining time-series visualizations using Large Language Models with
vision capabilities, enabling multimodal analysis of charts and signals.
"""

import base64
import os
from typing import Dict

from .llm_config import _get_llm


class LLMImageAnalyzer:
    """
    A class for analyzing CNC machine time-series charts using LLM vision models.
    """

    DEFAULT_TEMPERATURE = 0.3
    DEFAULT_MAX_TOKENS = 2000

    # Context describing the CNC machine and its signals
    CNC_PROCESS_CONTEXT = """
    **CNC MACHINE CONTEXT:**
    The data comes from an industrial CNC milling machine monitored during machining operations.
    Each workpiece is identified by a Work Order number (OF), e.g. OF10001, OF10002.
    Data is recorded as time-series signals sampled every 5 seconds across three sensor files.

    **MACHINE STATES:**
    - Operation_Mode: current machine mode (e.g. AUTO = running NC program, MDI = manual input, JOG = manual axis movement, IDLE = standby)
    - Operation_Status: current execution status (e.g. RUNNING, STOPPED, ALARM, WAITING)
    - Program_Name: name of the NC (G-code) program being executed
    - Program_Block_Number: current block/line number in the NC program (indicates machining progress)

    **TOOLING:**
    - Tool_Number: identifier of the active cutting tool (changes at tool change operations)
    - Head_Angular_On: angular milling head is active (1 = active, 0 = inactive)
    - Head_Auto_On: automatic head mode is engaged
    - Head_Boring_On: boring operation mode is active

    **VIBRATION / CHATTER DETECTION:**
    - Chatter_Detection_OnOff_X: chatter (unwanted vibration) detected on X axis (1 = detected)
    - Chatter_Detection_OnOff_Y: chatter detected on Y axis

    **NUMERIC SIGNALS (typical):**
    - Axis positions: X, Y, Z coordinates of the cutting tool (in mm)
    - Spindle speed: rotation speed of the cutting tool (in RPM)
    - Feed rate: tool advancement speed (in mm/min)
    - Axis loads / motor currents: mechanical load on each axis motor (in %)
    - Cutting force / torque estimates
    """

    CHART_READING_INSTRUCTIONS = """
    **CHART READING INSTRUCTIONS:**
    - X-axis: time progression (timestamps in HH:MM:SS or datetime format)
    - Y-axis: signal value — unit depends on the signal (mm for positions, RPM for speed, % for loads, 0/1 for binary states)
    - Different colored lines represent different CNC signals or axes
    - Flat segments indicate the machine is idle or holding position
    - Steep changes indicate rapid movements or mode transitions
    - Repeated patterns suggest repeating machining cycles (e.g. roughing passes)
    - Tool_Number steps indicate tool changes between operations
    - Program_Block_Number increasing linearly indicates continuous NC program execution
    """

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the LLM image analyzer.

        Args:
            temperature: Temperature for response generation (lower = more deterministic)
            max_tokens: Maximum tokens in the LLM response
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = _get_llm()  # fixed: was self._llm (attribute name mismatch)

    def _read_image_as_base64(self, image_path: str) -> str:
        """
        Read and encode an image file as a base64 string.

        Args:
            image_path: Path to the PNG image file

        Returns:
            Base64 encoded image string

        Raises:
            FileNotFoundError: If the image file does not exist
            IOError: If the file cannot be read
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Chart image not found at {image_path}")

        try:
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
            return base64.b64encode(image_data).decode()
        except IOError as e:
            raise IOError(f"Unable to read image file: {e}")

    def _create_vision_message(self, prompt: str, image_base64: str) -> Dict:
        """
        Create a multimodal message containing both text and image for the LLM.

        The LLM receives the image encoded as base64 alongside the text prompt,
        allowing it to analyze the chart visually and cross-reference with the
        provided numerical data summary.

        Args:
            prompt: The analysis prompt (text context + instructions)
            image_base64: Base64 encoded image

        Returns:
            Formatted message dictionary (OpenAI-compatible multimodal format)
        """
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                },
            ],
        }

    def _get_response_content(self, response) -> str:
        """
        Extract the text content from an LLM response object.

        Args:
            response: LLM response object

        Returns:
            Response content as a plain string
        """
        return response.content if hasattr(response, "content") else str(response)

    def _create_data_enhanced_prompt(self, process_data_summary: str) -> str:
        """
        Build the full analysis prompt combining CNC context, chart instructions,
        and the actual numerical data summary extracted from the workpiece files.

        Args:
            process_data_summary: Computed summary of the CNC signals
                                  (cycle count, durations, axis ranges, tool changes, etc.)

        Returns:
            Complete prompt string to send to the vision LLM
        """
        analysis_requirements = """
        **ANALYSIS REQUIREMENTS:**
        1. Base your analysis on BOTH the chart visualization AND the provided process data summary.
        2. Reference specific timestamps visible in the chart.
        3. Correlate chart signal patterns with the quantitative metrics in the data summary.
        4. Identify distinct machining phases (e.g. approach, roughing, finishing, tool change, retract).
        5. Use exact signal values — avoid generic statements.
        6. Detect anomalies: chatter events, unexpected idle periods, load spikes, or alarm states.
        """

        analysis_structure = """
        #### Machining Phase Identification
        Based on the chart patterns and the process data summary, identify and describe each
        distinct machining phase visible in the time window. Reference the timestamps and
        signal transitions that mark phase boundaries (e.g. tool change, spindle start/stop,
        program block progression, axis direction reversals).

        #### Technical Signal Analysis
        Provide a detailed analysis of the key CNC signals:
        - Axis positions (X, Y, Z): describe movement ranges, direction changes, and dwell periods.
        - Spindle speed: identify speed changes and their correlation with tool or operation changes.
        - Feed rate: note variations and their relation to roughing vs. finishing passes.
        - Axis loads / motor currents: identify overload events or unusual load patterns.
        - Tool_Number: list all tool changes and estimate the duration of each tool's use.
        - Chatter detection: report any chatter events on X or Y axes and when they occurred.

        #### Process Quality & Anomaly Detection
        Using both the chart and the data summary:
        - Identify any anomalies (unexpected stops, alarms, chatter, load spikes).
        - Assess process consistency: are machining cycles repeatable in duration and signal amplitude?
        - Note any idle periods longer than expected and their possible causes.
        - Highlight optimization opportunities (e.g. excessive rapid traverse, unnecessary dwells).

        **RESPONSE REQUIREMENTS:**
        - Use specific timestamps from the chart (e.g. "at 14:05:30").
        - Reference exact signal values (e.g. "X axis reached -245.3 mm at 14:07:12").
        - Use the exact metrics from the data summary (cycle count, durations, tool change timestamps).
        - Write for CNC machine operators and process engineers who need actionable, precise information.

        Start your response directly with "#### Machining Phase Identification" — no introductory phrases.
        """

        return f"""Analyze this CNC milling machine time-series chart together with the actual process data summary provided below.

        {analysis_requirements}

        {self.CHART_READING_INSTRUCTIONS}

        {self.CNC_PROCESS_CONTEXT}

        **ACTUAL PROCESS DATA SUMMARY:**
        {process_data_summary}

        **ANALYSIS TASK:**
        Using both the chart visualization and the process data above:
        1. Identify each machining phase and its time boundaries.
        2. Correlate chart signal patterns with the quantitative metrics provided.
        3. Reference exact timestamps and signal values from the chart.
        4. Use the cycle count, tool change events, and duration metrics from the data summary.
        5. Detect and explain any anomalies or quality concerns visible in the signals.
        6. Provide actionable observations for the machine operator.

        {analysis_structure}"""

    def analyze_with_data_and_image(
        self, image_path: str, process_data_summary: str
    ) -> str:
        """
        Generate a comprehensive CNC machining analysis combining a chart image
        and a numerical process data summary.

        Args:
            image_path: Path to the saved chart image (PNG format)
            process_data_summary: Computed summary of CNC signals for the workpiece

        Returns:
            Comprehensive text analysis combining visual and numerical insights
        """
        image_base64 = self._read_image_as_base64(image_path)
        prompt = self._create_data_enhanced_prompt(process_data_summary)
        message = self._create_vision_message(prompt, image_base64)
        response = self.llm.invoke([message])
        return self._get_response_content(response)

    def analyze_with_data_image_and_custom_prompt(
        self, image_path: str, process_data_summary: str, custom_prompt: str
    ) -> str:
        """
        Generate a CNC machining analysis combining a chart image, numerical data,
        and a custom user question or focus area.

        Args:
            image_path: Path to the saved chart image (PNG format)
            process_data_summary: Computed summary of CNC signals for the workpiece
            custom_prompt: Specific analysis request from the user
                           (e.g. "focus on chatter events", "explain the tool change at 14:07")

        Returns:
            Comprehensive analysis addressing both the standard structure and the custom request
        """
        image_base64 = self._read_image_as_base64(image_path)
        base_prompt = self._create_data_enhanced_prompt(process_data_summary)
        enhanced_prompt = f"""{base_prompt}

        **ADDITIONAL CUSTOM ANALYSIS REQUEST:**
        {custom_prompt.strip()}

        Please address this specific request while maintaining the structured analysis format above.
        Use both the chart visualization and the process data to answer thoroughly."""

        message = self._create_vision_message(enhanced_prompt, image_base64)
        response = self.llm.invoke([message])
        return self._get_response_content(response)


class ChartImageSaver:
    """
    A class for saving Plotly charts as high-quality PNG images.
    """

    DEFAULT_WIDTH = 900
    DEFAULT_HEIGHT = 600
    DEFAULT_SCALE = 1   # was 3 (4200×2700 px) → now 900×600, ~10× faster kaleido render
    DEFAULT_FORMAT = "png"
    DEFAULT_ENGINE = "kaleido"

    def __init__(
        self,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        scale: int = DEFAULT_SCALE,
        format: str = DEFAULT_FORMAT,
        engine: str = DEFAULT_ENGINE,
    ):
        self.width = width
        self.height = height
        self.scale = scale
        self.format = format
        self.engine = engine

    def _ensure_directory_exists(self, filepath: str) -> None:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

    def _configure_figure_for_export(self, fig):
        """Apply a clean white background for optimal PNG export."""
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="black"),
        )
        return fig

    def save_chart(self, fig, filepath: str) -> bool:
        """
        Save a Plotly figure as a high-quality PNG image.

        Args:
            fig: Plotly figure object
            filepath: Destination path for the PNG file

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            self._ensure_directory_exists(filepath)
            configured_fig = self._configure_figure_for_export(fig)
            configured_fig.write_image(
                filepath,
                format=self.format,
                width=self.width,
                height=self.height,
                scale=self.scale,
                engine=self.engine,
            )
            return True
        except Exception as e:
            print(f"Error saving chart: {e}")
            return False


def save_chart_as_image(fig, filepath: str) -> bool:
    """
    Save a Plotly figure as a high-quality PNG image.

    Args:
        fig: Plotly figure object
        filepath: Destination path for the PNG file

    Returns:
        True if saved successfully, False otherwise
    """
    saver = ChartImageSaver()
    return saver.save_chart(fig, filepath)


def generate_comprehensive_process_explanation(
    image_path: str,
    process_data_summary: str,
    custom_prompt: str = None,
) -> str:
    """
    Generate a comprehensive CNC machining explanation using image + data analysis.

    Always uses the combined data + image approach for maximum analytical depth.
    Optionally incorporates a custom user question for focused analysis.

    Args:
        image_path: Path to the saved chart image (PNG format)
        process_data_summary: Computed summary of CNC signals (required)
        custom_prompt: Optional custom analysis request from the user

    Returns:
        Comprehensive text explanation of the CNC machining session
    """
    analyzer = LLMImageAnalyzer(temperature=0.1, max_tokens=2000)

    if custom_prompt and custom_prompt.strip():
        print("Generating CNC analysis (data + image + custom prompt)...")
        return analyzer.analyze_with_data_image_and_custom_prompt(
            image_path, process_data_summary, custom_prompt
        )
    else:
        print("Generating CNC analysis (data + image)...")
        return analyzer.analyze_with_data_and_image(image_path, process_data_summary)
