Overview:
The code implements an inspection system using cameras (device00 and device01) to capture images for
inspection and analysis of parts specified by their coordinates (xy.csv files).
It utilizes OpenCV for image processing, TensorFlow for machine learning operations,
and PySimpleGUI for creating a graphical user interface (GUI).


It has following key parts

1. Initialization and Configuration
	i) The environment is setup in python 3.9.13, using latest python may cause trouble.
   	   Requirement files is updated as of July 2024.

	ii)GPU is configured to support memory growth


	iii)Camera Interaction:
				Imports and integrates the tisgrabber library (import utils.tisgrabber as 
				tis) for camera interaction.Defines callback functions 
				(frameReadyCallbackDevice00, frameReadyCallbackDevice01) to handle 
				image capture events triggered by the cameras 
				(hGrabberDevice00, hGrabberDevice01).

2. Image Processing Functions:

	i)base_crop_save(spec, part_names, deviceName):
				Processes images captured by cameras (device00 and device01).Cuts out
				specific parts (part_names) based on polygon coordinates (xy.csv) 
				Utilizes functions for polygon manipulation (loadPolyxy, shrink_poly) 
				masking (make_mask), cropping (part_crop), and stabilization 
				(stabilizer_method).
 
3. Testing and Analysis Functions:

	i)test(specRB, listPieceRB):
				Conducts testing on parts of image (specRB and specRC) captured by 
				the cameras.Evaluates testing results 
				(listTestResultRB and listTestResultRC) to determine if parts are 
				satisfactory (ok) or faulty (ng).Aggregates individual part results 
				to determine the overall inspection outcome (result).

	ii)getResultImg(img, dictGuixy, listPiece, listTestResult):

				Modifies captured images (img) based on testing results (listTestResult) 
				for each part (listPiece).Highlights failed parts by filling them with 
				red color ((0, 0, 255)).

4. GUI and User Interaction Graphical User Interface (GUI):

				Displays inspection results (imgResult) indicating system status 
				(imgStart.png, imgOk.png, imgNG.png, imgError.png).
				Presents specific part images (imgRB, imgRC) and text information 
				(TextSpec, TextSerial) related to the inspected parts.Allows user 
				interaction through a button (buttonStart) to initiate inspections 
				and potentially trigger camera captures.

5. System Control and Error Handling:

				Manages file paths (os, shutil) for storing captured images 
				(device00, device01) and other system operations.
				Implements error handling (if flgPicLoad == False) and timeout mechanisms
				(for i in range(11)) to ensure robustness during image acquisition and 
				processing.Properly stops camera operations (stop_camera) to release resources
 				after completing inspections.Closes the GUI window (window.close) to conclude 
				system operations.