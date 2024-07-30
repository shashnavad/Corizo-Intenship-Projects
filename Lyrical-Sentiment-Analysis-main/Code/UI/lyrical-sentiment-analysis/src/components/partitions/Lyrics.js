import { useContext, useRef } from "react";
import StateContext from "../../store/state-context";

import css from "./Lyrics.module.css";
import Button from "../ui/Button";

function escapeHtml(string) {
	var charMap = {
		'&': '&amp;',
		'<': '&lt;',
		'>': '&gt;',
		'"': '&quot;',
		"'": '&#39;',
		'/': '&#x2F;',
		'`': '&#x60;',
		'=': '&#x3D;'
	};
	return String(string).replace(/[&<>"'`=\\/]/g, function(s) {
		return charMap[s];
	});
}

async function runModel(model, lyrics) {
	return await fetch(`${window.location.protocol}//${window.location.host}/classify`, {
		method: 'POST',
		headers: {
			"Content-type": "application/json",
			"Accept": "application/json"
		},
		body: JSON.stringify({
			model: model,
			lyrics: lyrics
		})
	}).then(response => {
		if (!response.ok) {
			throw new Error('Network response was not ok');
		}
		return response.json();
	});
}

function Lyrics() {
	const lyricsInputRef = useRef();
	const stateContext = useContext(StateContext); 
	
	async function onClickHandler(model) {
		const inputtedLyrics = escapeHtml(lyricsInputRef.current.value).replace(",", "");
		stateContext.updateModel({
			lyrics: inputtedLyrics,
			model: model,
			isRunning: true,
			results: null
		});
		let results = await runModel(model, inputtedLyrics);
		stateContext.updateModel({
			isRunning: false,
			results: results
		})
	}

	const disabled = stateContext.isRunning;
	return (
		<div>
			<div className={css.title}>Lyrics</div>
			<br />
			<label htmlFor="lyrics">Please enter the lyrics you'd like to analyze below:</label>
			<textarea className={css.input} type="text" ref={lyricsInputRef} id="lyrics" name="lyrics" spellCheck={false} disabled={disabled}></textarea>
			<div className={css.actions}>
				<Button caption="Run Classical Model" id="Classical" disabled={disabled} onClick={onClickHandler} />
				<Button caption="Run Deep Learning Model" id="Deep Learning" disabled={disabled} onClick={onClickHandler} />
			</div>
		</div>
	);
}

export default Lyrics;