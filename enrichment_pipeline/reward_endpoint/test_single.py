import requests

# The API endpoint
urls = [
    "http://47.189.79.46:50159", # 3090s1
    "http://47.189.79.46:50108",
    "http://47.189.79.46:50193",
    "http://47.189.79.46:50060",
    # 'http://211.21.106.84:57414', # 3090s2
    # 'http://211.21.106.84:57515',
    # 'http://211.21.106.84:57298',
    # 'http://211.21.106.84:57445',
]

prompt = '''
Maryland Gov. Larry Hogan (R) upended the regional debate over Metro funding Monday by offering to give the transit system an extra $500 million over four years if Virginia, the District and the federal government each do the same.\n\nHogan's proposal, made in a letter delivered Monday morning to Virginia Gov. Terry McAuliffe (D) and D.C. Mayor Muriel E. Bowser (D), narrowed their differences over funding and appeared to increase chances that the region could agree on a plan to save the agency.\n\nBut it remained to be seen whether the other three parties \u2014 especially the federal government and Virginia \u2014 would go along. Some politicians grumbled that Hogan only made the proposal because he knew it was unlikely to be accepted, and a Metro board member predicted the federal government would balk.\n\nOverall, however, top Metro officials and other regional leaders praised Hogan for taking an important first step toward reaching consensus, while they warned that the plan falls short of a permanent solution.\n\nHogan's action marked a dramatic reversal from his position in a contentious, closed-door, regional summit two weeks ago. There, Hogan shocked McAuliffe and Bowser by saying Maryland would not give Metro any additional funds beyond what it already contributes each year.\n\n[Behind closed doors, region\u2019s leaders clashed sharply over Metro funding]\n\nOn Monday, Hogan reaffirmed the stance he took at the summit against new taxes to support Metro, and he complained that Maryland contributes more than its fair share to the struggling transit agency. But he took a new approach regarding more money.\n\n\"The needs of the Metro system are immediate and overwhelming,\" Hogan wrote in the detailed, four-page letter. \"Given the current crisis, the State of Maryland is prepared to invest an additional $500 million in increased Metro funding over the next four years if the Commonwealth of Virginia, the District of Columbia and the federal government all commit to do the same.\"\n\nHogan's about-face appeared prompted partly by intense criticism of his earlier opposition, officials said, both from other regional actors and from a strongly worded editorial in The Washington Post headlined, \"Larry Hogan to Metro: Drop Dead.\" Hogan is expected to seek reelection next year, and an anti-Metro stance could hurt him in vote-rich Montgomery and Prince George's counties.\n\nBut Hogan spokesman Doug Mayer suggested the governor had taken the adamant position at the summit as a bargaining ploy at the start of what he expected to be a prolonged process.\n\nHogan is \"always negotiating,\" Mayer said.\n\nMayer also rejected the idea that Hogan had altered his position, noting that the governor emphasized in the letter his previous stance that the federal government ought to contribute more to Metro. In proposing more money now from Maryland, Virginia and the District, Mayer said, the governor was expanding on his earlier strategy.\n\n[Maryland to get $900-million federal full funding agreement for Purple Line]\n\nRep. Gerald E. Connolly (D-Va.) welcomed Hogan's change of mind and said he believed it came in response to the backlash to Hogan's position at the summit\n\nSummarize the preceding context in 4 sentences. Do not try to create questions or answers for your summarization.\n\n

'''
completion = '''
Here is a summary of the given text in four sentences:\n\n1. Maryland Governor Larry Hogan has offered to provide an additional $500 million in funding over four years to help save the Metrorail system if Virginia, DC, and the federal government match this amount.\n2. This offer represents a significant shift from Hogan's previously stated position at a recent regional summit where he refused to consider adding funds beyond those currently contributed annually by Maryland.
'''
# Your data
data = {
    "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@",  # Your authentication token
    "prompt": prompt,
    "completions": [completion, completion],
}

for url in urls:
    for i in range(1):
        response = requests.post(url, json=data)
        print(response.json())