import google.generativeai as genai
import os
import json
import time
import moviepy.editor as mp  



def setup_api_key():
    """
    Retrieves the Gemini API key from an environment variable.
    This is more secure than hardcoding the key.
    """
    api_key = "AIzaSyCqZ6Yby1dbdh-tr5XtXEYdK7z-ywzbubs" 
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set the variable (e.g., export GEMINI_API_KEY='your_key_here')")
        return None

    genai.configure(api_key=api_key)
    return True


def upload_file_with_retry(file_path, max_retries=3):
    """
    Uploads a file to the Gemini API File Service with retry logic.
    It now also polls and waits for the file to become ACTIVE.
    """
    print(f"Uploading file: {file_path}...")


    try:
        FileState = genai.types.File.State
    except AttributeError as e:
        
        
        class FileState:
            PROCESSING = 1
            ACTIVE = 2
            FAILED = 3
        print("Warning: Falling back to assumed integer states (1=PROCESSING, 2=ACTIVE).")

    retry_count = 0
    while retry_count < max_retries:
        try:
            myfile = genai.upload_file(file_path)
            print(f"File uploaded successfully. URI: {myfile.uri}, Name: {myfile.name}")

            
            print(f"Waiting for file '{myfile.name}' to become ACTIVE...")
            print(f"Initial file state: {myfile.state}")  

            while myfile.state == FileState.PROCESSING:
                print("... file state is PROCESSING, waiting 5 seconds...")
                time.sleep(5)  
                myfile = genai.get_file(myfile.name)
                print(f"... new state: {myfile.state}")

            if myfile.state == FileState.ACTIVE:
                print("File is ACTIVE and ready for use.")
                return myfile
            else:
                print(f"Error: File '{myfile.name}' finished in state '{myfile.state}', not ACTIVE.")
                if myfile.state == FileState.FAILED:
                    print("File processing FAILED.")
                else:
                    print(f"File is in an unexpected state: {myfile.state}")

                try:
                    genai.delete_file(myfile.name)
                    print(f"Deleted file in non-ACTIVE state: {myfile.name}")
                except Exception as del_e:
                    print(f"Error deleting file in non-ACTIVE state: {del_e}")
                return None 

        except Exception as e:
            retry_count += 1
            print(f"Error uploading file (Attempt {retry_count}/{max_retries}): {e}")
            if "500" in str(e):  
                time.sleep(2**retry_count)  
            else:
                break  
    print(f"Failed to upload file {file_path} after {max_retries} attempts.")
    return None


def safe_json_parse(text_response):
    """
    Tries to parse a JSON string, cleaning it up if it's wrapped in backticks.
    """
    try:
        if text_response.strip().startswith("```json"):
            text_response = text_response.strip()[7:-3].strip()
        elif text_response.strip().startswith("```"):
            text_response = text_response.strip()[3:-3].strip()

        return json.loads(text_response)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from response.")
        print(f"Response was: {text_response}")
        return None


def analyze_audio(file_path):
    """
    Analyzes an audio file for vocal cues.
    """
    print("\n--- Starting Audio Analysis ---")
    myfile = upload_file_with_retry(file_path)
    if not myfile:
        return None

    prompt = """
    Analyze the speaker's tone in this audio file. 
    Listen for vocal indicators often associated with stress or deception, 
    such as nervousness, hesitation, unnatural pauses, filler words (like 'um', 'ah'), 
    or significant changes in pitch or speed.
    
    Provide your analysis, a "deception_score" from 0 (completely truthful sounding) 
    to 10 (highly deceptive sounding) based *only* on these vocal cues, and a
    list of specific proxies you detected.

    Respond with ONLY a valid JSON object in the following format:
    {
      "analysis": "Your detailed analysis of the speaker's tone...",
      "detected_proxies": ["list", "of", "detected", "vocal", "proxies", "like", "hesitation"],
      "deception_score": <number_from_0_to_10>
    }
    """

    try:
        print("Generating audio content analysis...")
        model = genai.GenerativeModel("gemini-2.5-flash") 
        response = model.generate_content([prompt, myfile])

        print(f"Deleting uploaded file: {myfile.name}...")
        genai.delete_file(myfile.name)
        print("Audio file deleted.")

        return safe_json_parse(response.text)

    except Exception as e:
        print(f"Error during audio analysis: {e}")
        try:
            genai.delete_file(myfile.name)
            print("Audio file deleted after error.")
        except Exception as del_e:
            print(f"Error deleting file after analysis error: {del_e}")
        return None


def analyze_video(file_path):
    """
    Analyzes a video file for visual cues.
    """
    print("\n--- Starting Video Analysis ---")
    myfile = upload_file_with_retry(file_path)
    if not myfile:
        return None

    prompt = """
    Analyze the speaker's visual cues in this video file. 
    Look for non-verbal indicators often associated with stress or deception, 
    such as fidgeting, lack of eye contact, gaze aversion, shifting, 
    unnatural body language, self-touching (like touching face or neck), 
    or micro-expressions.
    
    Provide your analysis, a "deception_score" from 0 (completely truthful looking) 
    to 10 (highly deceptive looking) based *only* on these visual cues, and a
    list of specific proxies you detected.

    Respond with ONLY a valid JSON object in the following format:
    {
      "analysis": "Your detailed analysis of the speaker's body language...",
      "detected_proxies": ["list", "of", "detected", "visual", "proxies", "like", "fidgeting"],
      "deception_score": <number_from_0_to_10>
    }
    """

    try:
        print("Generating video content analysis...")
        model = genai.GenerativeModel("gemini-2.5-flash") 
        response = model.generate_content([prompt, myfile])

        
        print(f"Deleting uploaded file: {myfile.name}...")
        genai.delete_file(myfile.name)
        print("Video file deleted.")

        return safe_json_parse(response.text)

    except Exception as e:
        print(f"Error during video analysis: {e}")
        
        try:
            genai.delete_file(myfile.name)
            print("Video file deleted after error.")
        except Exception as del_e:
            print(f"Error deleting file after analysis error: {del_e}")
        return None


def get_final_score(audio_result, video_result):
    """
    Combines audio and video analysis into a final "truth score".
    This is a text-only call, no file upload needed.
    """
    print("\n--- Starting Combined Analysis ---")

    
    audio_analysis = audio_result.get("analysis", "No audio analysis available.")
    video_analysis = video_result.get("analysis", "No video analysis available.")

    audio_score = audio_result.get("deception_score", "N/A")
    video_score = video_result.get("deception_score", "N/A")

    
    audio_proxies = audio_result.get("detected_proxies", [])
    video_proxies = video_result.get("detected_proxies", [])

    prompt = f"""
    I have two separate analyses of a person speaking, one from their audio 
    and one from their video. I need you to act as a final arbiter and combine 
    these findings into a single, final "truth_score" from 0 (very truthful) 
    to 10 (very deceptive).

    Audio Analysis:
    - Analysis Text: "{audio_analysis}"
    - Reported Audio Score (0-10): {audio_score}
    - Detected Vocal Proxies: {audio_proxies}

    Video Analysis:
    - Analysis Text: "{video_analysis}"
    - Reported Video Score (0-10): {video_score}
    - Detected Visual Proxies: {video_proxies}

    Based on *both* of these analyses and the specific proxies detected, 
    provide a final, synthesized summary.
    If the analyses contradict (e.g., audio sounds calm but video is very 
    nervous), please note that. Also, comment on whether the detected
    proxies from audio and video support or contradict each other.
    
    Conclude with a final "truth_score".

    Respond with ONLY a valid JSON object in the following format:
    {{
      "summary": "Your combined analysis summary, noting contradictions and proxy correlations...",
      "truth_score": <number_from_0_to_10>
    }}
    """

    try:
        print("Generating final combined score...")
        model = genai.GenerativeModel("gemini-2.5-flash") 
        response = model.generate_content(prompt)
        return safe_json_parse(response.text)

    except Exception as e:
        print(f"Error during final analysis: {e}")
        return None


def extract_audio(video_path):
    """
    Extracts the audio from a video file and saves it as a temporary mp3 file.
    Returns the path to the extracted audio file, or None if it fails.
    """
    print(f"\n--- Extracting Audio from {video_path} ---")
    extracted_audio_path = None
    video_clip = None
    try:
        video_clip = mp.VideoFileClip(video_path)
        if video_clip.audio is None:
            print(f"Error: The video file {video_path} has no audio track.")
            return None

        
        base_name = os.path.basename(video_path)
        file_name_without_ext = os.path.splitext(base_name)[0]
        extracted_audio_path = f"temp_audio_for_{file_name_without_ext}.mp3"

        
        print(f"Writing temporary audio to {extracted_audio_path}...")
        
        video_clip.audio.write_audiofile(extracted_audio_path, logger=None)
        
        print(f"Audio extracted successfully.")
        return extracted_audio_path

    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None
        
    finally:
        
        if video_clip:
            video_clip.close()


def main():
    """
    Main function to run the analysis pipeline.
    """
    if not setup_api_key():
        return
    
    input_video_path = r"C:\Users\naman\Desktop\lie\test\video.mp4"
    if not os.path.exists(input_video_path):
        print(f"Error: Video file not found at {input_video_path}")
        print("Please update the 'input_video_path' variable in the script.")
        return

    extracted_audio_path = None  
    try:
        
        extracted_audio_path = extract_audio(input_video_path)
        if not extracted_audio_path:
            print("Aborting analysis as audio could not be extracted or is missing.")
            return

        
        audio_result = analyze_audio(extracted_audio_path)

        
        video_result = analyze_video(input_video_path)

        
        if audio_result and video_result:
            final_result = get_final_score(audio_result, video_result)

            print("\n\n============ FINAL REPORT ============")

            print("\n--- Audio Analysis Result ---")
            print(json.dumps(audio_result, indent=2))

            print("\n--- Video Analysis Result ---")
            print(json.dumps(video_result, indent=2))

            print("\n--- Combined Final Analysis ---")
            print(json.dumps(final_result, indent=2))

            print("\n========================================")

            if final_result:
                score = final_result.get('truth_score', 'N/A')
                print(f"\n>>>> Final Truth Score (0-10): {score} <<<<")

        else:
            print("\n--- Analysis Incomplete ---")
            print("One or both of the analysis steps failed. Cannot generate final score.")
            if audio_result:
                print("\nAudio Analysis (Partial):")
                print(json.dumps(audio_result, indent=2))
            if video_result:
                print("\nVideo Analysis (Partial):")
                print(json.dumps(video_result, indent=2))
                
    except Exception as e:
        print(f"An unexpected error occurred in main: {e}")

    finally:
        
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            print(f"\nCleaning up temporary audio file: {extracted_audio_path}")
            try:
                os.remove(extracted_audio_path)
                print("Cleanup successful.")
            except Exception as e:
                
                print(f"Error cleaning up file {extracted_audio_path}: {e}")
                print("You may need to delete this file manually.")


if __name__ == "__main__":
    main()