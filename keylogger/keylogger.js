// PROBLEMS WITH KEY LOGGER
// doesnt support realsing 2nd before 1st character
// 

var timer = {};
var SPACE_KEY = 32;
var ENTER_KEY = 13;
var BACKSPACE_KEY = 8;
var keys = [];
var i = 0;

var letancies = [];
var password = [];
var passwordStr = "";
var password_latency = [];
var inputing = true;
/// write to file
var txtFile = "log.txt";
var file = new File([""], txtFile);
function reset() {
	timer = {};
	keys = [];
	i = 0;
	letancies = [];
	password = [];
	passwordStr = "";
	password_latency = [];
	inputing = true;
	document.getElementById("count").innerHTML = 0;
	document.getElementById("pass").innerHTML = "--";

}
function download() {
    var textToSaveAsBlob = new Blob([passwordStr + "\r\n" + letancies.join("\r\n")], {type:"text/plain"});
    var textToSaveAsURL = window.URL.createObjectURL(textToSaveAsBlob);
    var fileNameToSaveAs = "log.txt";

    var downloadLink = document.createElement("a");
    downloadLink.download = fileNameToSaveAs;
    downloadLink.innerHTML = "Download File";
    downloadLink.href = textToSaveAsURL;
    downloadLink.onclick = destroyClickedElement;
    downloadLink.style.display = "none";
    document.body.appendChild(downloadLink);
 
    downloadLink.click();
}

function destroyClickedElement(event)
{
    document.body.removeChild(event.target);
}

function keydown(event) {
	var key = event.which || event.keyCode;
	if(!timer[key]) {
		timer[key] = Math.round(10*performance.now())/10;
	}
}

function keyup(event) {
	
	var key = event.which || event.keyCode;
	var latency = Math.round(10*(performance.now() - timer[key]))/10;
	keys.push({"key": key, "time": timer[key], "latency": latency});
	if(inputing) {
		console.log("pressed "+key+"-"+String.fromCharCode(key)+" for "+latency+" seconds.");
	}
	if(key == ENTER_KEY) {
		
		//console.log("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

		if(inputing) {
			console.log("~~~ PASSWORD: "+password);
			password_latency = [];
			inputing = false;
			document.getElementById("pass").innerHTML = document.getElementById("Text1").value;
			passwordStr = document.getElementById("Text1").value;
		}
		document.getElementById("Text1").value = "";
		if(password_latency.length == password.length * 2 - 1) {
			letancies.push(password_latency);
			document.getElementById("count").innerHTML = letancies.length
			console.log("~~~ PUSHED");
			//console.log("~~~ LETANCY: "+password_latency);
			//console.log("~~~ LETANCIES: "+letancies);
		}
		password_latency = [];
		i = 0;
	}
	else if(inputing) {
		password.push(key);
		passwordStr += String.fromCharCode(key);
	}

	if(key == BACKSPACE_KEY || i > password.length || key != password[i]) {
		if(key != ENTER_KEY)
			console.log("~~~ DUMPED LATENCY");
		password_latency = [];
		i = 0;
	}
	else {
		if(i > 0) {
			var time_between = timer[key] - keys[keys.length - 2].time - keys[keys.length - 2].latency;
			password_latency.push(Math.round(10*time_between)/10);
		}
		//console.log("~~~ PUSHING LATENCY");
		password_latency.push(latency);
		i++;
	}	
	timer[key] = 0;
	
}