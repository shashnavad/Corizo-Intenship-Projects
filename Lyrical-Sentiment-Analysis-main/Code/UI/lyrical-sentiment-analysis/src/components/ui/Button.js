import css from "./Button.module.css"

function Button(props) {
	return <button className={css.btn} disabled={props.disabled === null ? true : props.disabled} onClick={() => props.onClick(props.id)}>{props.caption}</button>
}

export default Button;