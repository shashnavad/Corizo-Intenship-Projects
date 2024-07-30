import { useContext, useRef } from "react";
import StateContext from "../../store/state-context";

import css from "./Classical.module.css";

function Classical() {
	const stateContext = useContext(StateContext); 
	
	return (
		<div>
			<div className={css.title}>Classical Model Results</div>
			<br />
			<p>Result: {stateContext.results}</p>
		</div>
	);
}

export default Classical;