var timer = {};
var SPACE_KEY = 32;
var ENTER_KEY = 13;
var BACKSPACE_KEY = 8;
var keys = [];
var i = 0;
S

var letancies = [];
var password = [];
var password_latency = [];
var inputing = true;

function keydown(event) {
	var key = event.which || event.keyCode;
	if(!timer[key]) {
		timer[key] = performance.now();
	}
}

function keyup(event) {
	
	var key = event.which || event.keyCode;
	var latency = performance.now() - timer[key];
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
		}
		if(password_latency.length == password.length * 2 - 1) {
			letancies.push(password_latency);
			console.log("~~~ PUSHED");
			//console.log("~~~ LETANCY: "+password_latency);
			//console.log("~~~ LETANCIES: "+letancies);
		}
		password_latency = [];
		i = 0;
	}
	else if(inputing) {
		password.push(key);
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
			password_latency.push(time_between);
		}
		//console.log("~~~ PUSHING LATENCY");
		password_latency.push(latency);
		i++;
	}
	

	/*
	if(key == SPACE_KEY) {
		for(var i = 2; i < 10; i++) {
			if(keys[keys.length - i].key == SPACE_KEY) {
				var word = "";
				var time = keys[keys.length - 2].time + keys[keys.length - 2].letancy - keys[keys.length - i + 1].time;
				for(var j = keys.length-i+1; j < keys.length-1; j++) {
					word += String.fromCharCode(keys[j].key).toLowerCase();
				}
				console.log("word - "+word);
				if(words[word]) {
					words[word].push(time);
				}
				break;
			}
		}a	e
	*/

	
	timer[key] = 0;
	
}