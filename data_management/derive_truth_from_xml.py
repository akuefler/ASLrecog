import csv
import xml.etree.ElementTree as ET
import os
import re

def get_scaling_factor(meta_knowledge):
    return meta_knowledge["msEnd"] - meta_knowledge["msStart"] + 1

def convert_to_frames(value, scaling_factor, num_frames):
    return int(round(float(value) / scaling_factor * (num_frames - 1)) + 1)

def convert_to_ms(value, offset, scaling_factor, num_frames):
    return int(round(float(value - 1 + offset) / (num_frames - 1) * scaling_factor))
    
def print_coding_scheme(coding_scheme):
    for datatype in coding_scheme:
        print datatype + ", " + coding_scheme[datatype]["name"]
        for value in coding_scheme[datatype]:
            if value != "name":
                print "    " + value, coding_scheme[datatype][value]

# coding_scheme format:
# field id1 -> {name: 'label for field 1', 1: descriptionOfMeaningOfOneInField1, 2: descriptionOfMeaningOfTwoInField1}
# field id2 -> {name: 'label for field 2', 1: descriptionOfMeaningOfOneInField2, 2: descriptionOfMeaningOfTwoInField2}
def populate_coding_scheme(root):
    coding_scheme = {} 
    for record in root:
        if (record.tag == "CODING-SCHEME"):
            for field in record:
                assert(field.attrib.get('ID') not in coding_scheme)
                coding_scheme[field.attrib.get('ID')] = {}
                coding_scheme[field.attrib.get('ID')]["name"] = field.attrib.get('NAME')
                for value in field:
                    coding_scheme[field.attrib.get('ID')][value.attrib.get('ID')] = value.attrib.get('NAME')
    return coding_scheme
   
def parse_element(elem, codebook):
    if ("VID" in elem.attrib):
        assert(not elem.text)
        label = "(" + codebook[elem.attrib.get("VID")] + ")"
        start = (int)(elem.attrib.get("S"))
        end = (int)(elem.attrib.get("E"))
    else:
        label = elem.text
        start = (int)(elem.attrib.get("S"))
        end = (int)(elem.attrib.get("E"))
    return (label, start, end)

# TODO: make more efficient. we can store some of this knowledge.
def get_identifier_for(coding_scheme, phrase):
    identifier = None
    for item in coding_scheme:
        if "name" in coding_scheme[item] and coding_scheme[item]["name"] == phrase:
            identifier = item
            break
    if not identifier:
        print phrase
        print coding_scheme
        assert identifier
    return identifier

def is_gloss_but_not_english_translation(coding_scheme, machine_annotation_type):
    return is_gloss(machine_annotation_type) and not is_english_translation(coding_scheme, machine_annotation_type)

def is_gloss(machine_annotation_type):
    return int(machine_annotation_type) >= 10000

def is_english_translation(coding_scheme, machine_datatype):
    english_identifier = get_identifier_for(coding_scheme, "English translation")
    return machine_datatype == english_identifier

def is_main_gloss(coding_scheme, machine_datatype):
    gloss_identifier = get_identifier_for(coding_scheme, "main gloss")
    return machine_datatype == gloss_identifier

# Maintain an ongoing gloss for each millisecond, separated
# by utter_type (e.g., 'main gloss' or 'literal translation'),
# inside of the utter_rep representation of the utterance.
# Does so by iterating over all the milliseconds between start
# and end and appending the new parameter label to the existing label.
# Unlike the non-gloss items, we perform the expansion into each
# millisecond early for the gloss items so that we can maintain
# proper string-appending state (it's less easy to push the 
# expansion process to file writing time for the strings).
def append_gloss_to_all_intervening_milliseconds(utter_rep, utter_type, label, start, end):
    for msLabel in range(start, end+1):
        if utter_type not in utter_rep:
            utter_rep[utter_type] = {}
        if msLabel not in utter_rep[utter_type]:
            utter_rep[utter_type][msLabel] = ""  
        existingLabel = utter_rep[utter_type][msLabel]  
        utter_rep[utter_type][msLabel] = " ".join([existingLabel, label])
    return utter_rep

# Small form representation of an utterance:
#   "type -> (value)": {start: #, end: #} 
# It is "small" because we don't expand all the intervening milliseconds
# between start and end
def store_small_form(utter_rep, human_annotation_type, label, start, end):
    utter_rep[human_annotation_type + " => " + label] = {}
    utter_rep[human_annotation_type + " => " + label]["start"] = start
    utter_rep[human_annotation_type + " => " + label]["end"] = end
    return utter_rep

# At the global level, populate the meta knowledge about this
# track with its (a) English translation, (b) start ms, (c) end ms.
# This helps with converting to frames.
def populate_meta_knowledge(of_interest, filename, their_id, label, start, end):
    meta_knowledge = of_interest[filename][their_id]
    
    # Make sure we don't overwrite any existing data
    assert ("msStart" not in meta_knowledge)
    assert ("msEnd" not in meta_knowledge)
    assert ("english" not in meta_knowledge)
    
    # Update the global info for this track
    meta_knowledge["msStart"] = start
    meta_knowledge["msEnd"] = end
    meta_knowledge["english"] = label
    
    return of_interest

# Using the coding scheme, parse the XML and store in our internal
# format, in which glosses have been expanded to each intervening
# millisecond and which glosses occur at that time, and the 
# non-glosses have a "type -> (value)": {start: #, end: #} format. 
# Example output utterance representation:
# {
#     'eye brows => (lowered lid)': {'start': 34, 'end': 234}, 
#     'main gloss': {300: ' fs-JOHN', 301: ' fs-JOHN', 302: ' fs-JOHN', 303: ' fs-JOHN',    ..... }
# }
def populate_utterance_representation(utterance, coding_scheme, of_interest, filename, their_id):
    utter_rep = {}
    for track in utterance.findall("SEGMENT/TRACK"):
        machine_annotation_type = track.attrib.get("FID")
        human_annotation_type = coding_scheme[machine_annotation_type]["name"]
        for elem in track:
            (annotation_value, start, end) = parse_element(elem, coding_scheme[machine_annotation_type])

            if is_gloss_but_not_english_translation(coding_scheme, machine_annotation_type):
                utter_rep = append_gloss_to_all_intervening_milliseconds(utter_rep, human_annotation_type, annotation_value, start, end)
            else:
                utter_rep = store_small_form(utter_rep, human_annotation_type, annotation_value, start, end)

            if is_english_translation(coding_scheme, machine_annotation_type):
                populate_meta_knowledge(of_interest, filename, their_id, annotation_value, start, end)
        
    return utter_rep

def write_outputs(path, fn, utter_rep, meta_knowledge):
    truth_fn = path + fn + "_truth.csv"
    overview_fn = path + "overview.csv"
    
    write_truth_csv(truth_fn, utter_rep, meta_knowledge)
    append_to_overview(overview_fn, utter_rep, fn, meta_knowledge)

def write_truth_csv(filename, utter_rep, meta_knowledge):
    with open(filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for token in utter_rep:
            scaling = get_scaling_factor(meta_knowledge)
            nframes = meta_knowledge["frames"]
            if "=>" in token:
                start = utter_rep[token]["start"]
                end = utter_rep[token]["end"]

                startFrame = convert_to_frames(start, scaling, nframes)
                endFrame = convert_to_frames(end, scaling, nframes)

                #print u"%6d-%6d %s" % (startFrame, endFrame, token)

                writer.writerow([token] + \
                                    ['0'] * (startFrame - 1) + \
                                    ['1'] * (endFrame - startFrame + 1) + \
                                    ['0'] * (nframes - endFrame))
            else:
                resultStr = []
                resultStr += [token]
                for i in range(1, nframes + 1):
                    frameIdx = convert_to_ms(i, 0.5, scaling, nframes)
                    if frameIdx in utter_rep[token]:
                        resultStr += [utter_rep[token][frameIdx]]
                    else:
                        resultStr += ['']
                    #TODO currently just grabs on the frames; can improve this to get the union 
                    # of all tokens in the gloss between the ms that corresponds to start/end of frame
                    # For instance, BOOK in ncslgr10a-002-fs-JOHN_FINISH_READ_2_BOOK_ is listed
                    # as only one frame, when really it should be 2 or 3
                writer.writerow(resultStr)

# Given meta knowledge and an utterance, return a tuple indicating
# the start and end frame for a wh-word
def get_start_and_end_of_wh(meta_knowledge, utter_rep):       
    scaling = get_scaling_factor(meta_knowledge)
    nframes = meta_knowledge["frames"]
    
    item = utter_rep["POS => (Wh-word)"]
    startFrame = convert_to_frames(item["start"], scaling, nframes)
    endFrame = convert_to_frames(item["end"], scaling, nframes)
    
    return startFrame, endFrame 
                
def append_to_overview(overview_filename, utter_rep, truth_fn, meta_knowledge):
    whStart, whEnd = get_start_and_end_of_wh(meta_knowledge, utter_rep)
    
    with open(overview_filename, "ab") as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([truth_fn] + 
                        [meta_knowledge["english"]] + 
                        [meta_knowledge["frames"]] + 
                        [whStart] + 
                        [whEnd] + 
                        [meta_knowledge["msStart"]] + 
                        [meta_knowledge["msEnd"]] + 
                        [meta_knowledge["fnOurs"]])

def get_utterance_to_media(utterance, coding_scheme):
    identifier = utterance.attrib.get("ID")
    
    listOfMedia = []
    for media in utterance.findall("MEDIA-REF"):
        listOfMedia.append(media.attrib.get("ID"))
    
    trackLabel = [];
    for track in utterance.findall("SEGMENT/TRACK[@FID=\"10000\"]"):
        machine_annotation_type = track.attrib.get("FID")
        human_annotation_type = coding_scheme[machine_annotation_type]["name"]
        for elem in track:
            (annotation_value, start, end) = parse_element(elem, coding_scheme[machine_annotation_type])
            trackLabel.append(annotation_value)
    
    return identifier, "_".join(trackLabel), listOfMedia


def create_our_filenames_single_file(xmlFilename, utterances):
    our_fn_to_theirs = {}
    for u in utterances:
        # a completist "real" filename
        #our_fn = "%s-%03d-%s" % (xmlFilename.split(".xml")[0], int(u[0]) + 1, u[1]) 

        # an abbreviated filename that's easier to match on
        our_fn = OUR_FORMAT_STR % (xmlFilename.split(".xml")[0], convert_their_id_to_ours(int(u[0])))
        
        our_fn_to_theirs[our_fn] = u[2]      
    return our_fn_to_theirs

def create_our_filenames_all_files(xmlFilePath, xmlFiles):
    our_fn_to_theirs = {}
    for fn in xmlFiles:
        utterances = []

        tree = ET.parse(xmlFilePath + fn) 
        root = tree.getroot()

        # Get the utterances
        coding_scheme = populate_coding_scheme(root)
        for utterance in root.findall(".//UTTERANCE"):
            (identifier, gloss, arrayOfMedia) = get_utterance_to_media(utterance, coding_scheme)
            utterances.append((identifier, gloss, arrayOfMedia))
        our_fn = create_our_filenames_single_file(fn, utterances)
            
        # Create our filenames from them
        our_fn_to_theirs.update(our_fn)
        
    return our_fn_to_theirs

def print_our_fn_to_theirs(our_fn_to_theirs):
    for key in sorted(our_fn_to_theirs, key=str.lower):
        print key, our_fn_to_theirs[key]

def get_normed_filename(string):
    pattern = re.compile("[\w -]+-\d\d\d-")
    
    edges = pattern.search(string).span()
    start = edges[0]
    end = edges[1] - 1
    
    return string[start:end]
        
# Given a filename containing frame counts in which a line
# containing a filename is followed by a line containing
# the number of frames (as produced by getFrames.sh),
# returns a map of "filename-###" to an integer number of frames
def get_frame_counts(countfilename):
    fn_to_count = {}
    with open(countfilename, 'r') as countfile:
        lines = countfile.readlines() #currently small enough that this is reasonable
        for i in range(0, len(lines),2):
            filename = get_normed_filename(lines[i].strip())
            count = int(lines[i+1].strip())
            fn_to_count[filename] = count
    return fn_to_count

def get_filename_identifiers_of_interest(path_to_xml_annotations, path_to_frame_count_txt, path_to_video_folders):
    of_interest = {}

    # Get the original formats
    xml_file_path = path_to_xml_annotations
    xml_files = [f for f in os.listdir(xml_file_path) if os.path.isfile(os.path.join(xml_file_path,f)) ]
    our_fn_to_theirs = create_our_filenames_all_files(xml_file_path, xml_files)
    
    # Get the frame counts
    fn_to_frame_count = get_frame_counts(path_to_frame_count_txt)

    # Match to our video files
    video_file_path = path_to_video_folders
    video_files = [f for f in os.listdir(video_file_path) if os.path.isdir(os.path.join(video_file_path, f))]
    for full_filename in video_files:
        our_normalized_video_file = get_normed_filename(full_filename)
        
        xml_alone = our_normalized_video_file[:-4]
        id_alone = int(our_normalized_video_file[-3:])
        their_id = convert_our_id_to_theirs(id_alone)
        
        if (our_normalized_video_file in our_fn_to_theirs):
            frames = fn_to_frame_count[our_normalized_video_file]

            if xml_alone not in of_interest:
                of_interest[xml_alone] = {}
            of_interest[xml_alone][their_id] = {}
            of_interest[xml_alone][their_id]["fnOurs"] = full_filename
            of_interest[xml_alone][their_id]["frames"] = frames
        else:
            print "Does not exist!! " + our_normalized_video_file

    return of_interest

def convert_our_id_to_theirs(val):
    return val - 1

def convert_their_id_to_ours(val):
    return val + 1



OUR_FORMAT_STR = "%s-%03d"
    
# Assumes paths to folders end in /
path_to_xml_annotations = 'data/annotations/ncslgr-xml/'
path_to_frame_count_txt = 'data/truth/frameCounts.txt'
path_to_video_folders = 'data/intraface'
output_truth_path = 'data/'

of_interest = get_filename_identifiers_of_interest(path_to_xml_annotations, path_to_frame_count_txt, path_to_video_folders)

files = [ f for f in os.listdir(path_to_xml_annotations) if os.path.isfile(os.path.join(path_to_xml_annotations,f)) ]
for fn in files: #["ncslgr10a.xml"]: #files
    print fn + "..."
    tree = ET.parse(path_to_xml_annotations+fn) 
    root = tree.getroot()
    coding_scheme = populate_coding_scheme(root)
    
    fn_without_xml, file_extension = os.path.splitext(fn)
    if (fn_without_xml in of_interest):
        for their_id in of_interest[fn_without_xml].keys():
            for utterance in root.findall(".//UTTERANCE[@ID='"+str(their_id)+"']"):
                utter_rep = populate_utterance_representation(utterance, coding_scheme, of_interest, \
                                                              fn_without_xml, their_id)

                meta_knowledge = of_interest[fn_without_xml][their_id]
                write_outputs(output_truth_path, OUR_FORMAT_STR % (fn_without_xml, convert_their_id_to_ours(their_id)), \
                              utter_rep, meta_knowledge)
    else:
        print "--------> We don't have any videos related to %s" % fn
                
print "Done"