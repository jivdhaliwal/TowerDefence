package towerdefence.engine.entity;

import java.util.ArrayList;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.newdawn.slick.Color;
import org.newdawn.slick.GameContainer;
import org.newdawn.slick.Graphics;
import org.newdawn.slick.Input;
import org.newdawn.slick.SlickException;
import org.newdawn.slick.UnicodeFont;
import org.newdawn.slick.font.effects.ColorEffect;
import org.newdawn.slick.geom.Circle;
import org.newdawn.slick.geom.Vector2f;
import org.newdawn.slick.state.StateBasedGame;
import towerdefence.GameplayState;
import towerdefence.engine.Player;
import towerdefence.engine.ResourceManager;
import towerdefence.engine.Settings;
import towerdefence.engine.component.Component;
import towerdefence.engine.component.RenderComponent;

/**
 *
 * Tower - An entity that deals with distance checking against critters
 * and shoots critters
 * 
 *
 * @author Jiv Dhaliwal <jivdhaliwal@gmail.com>
 */
public class Tower extends Entity {
    
        // Tower types
    public final static int NORMAL = 0;
    public final static int FIRE = 1;
    public final static int ICE = 2;
    public final static int BULLET = 3;
    public final static int ROCKET = 4;

    ArrayList<Critter> critterList = null;
    private Critter targetCritter = null;

    private float range;
    float damagePerSec;
    private int type;

    private int shootingCounter;

    private boolean isPlaced=true;
    private boolean active;
    
    private Circle circle=null;
    private int mouseXTile;
    private int mouseYTile;
    
    private final UnicodeFont unicodeFont;
    
    String costOverlay;
    String overlay;
    String iceDetails = "150% damage\nto fire types";
    String fireDetails = "200% damage\nto ice types";
    String normalDetails = "85% damage\nto fire \nand ice types";
    


    public Tower(String id, boolean isActive) throws SlickException{
        super(id);

        this.rotation=0;
        this.active = isActive;
        
        unicodeFont = new UnicodeFont("fonts/pf_tempesta_seven.ttf", 8, false, false);
        unicodeFont.getEffects().add(new ColorEffect(java.awt.Color.white));
        
    }

    /*
     * Update the list of critters the tower iterates through to find a target
     */
    public void updateCritterList(ArrayList<Critter> critterList) {
        this.critterList = critterList;
    }

    private void renderTextOverlay(Graphics gr) {
        String tempOverlay="";
        String towerDetail="";
        
        if(type==FIRE) {
            towerDetail=fireDetails;
        } else if(type==ICE) {
            towerDetail=iceDetails;
        } else if(type==NORMAL) {
        	towerDetail=normalDetails;
        }
        
        if(active) {
            tempOverlay = "Sale Value : $"+
                    (Player.getInstance().getTowerCost(type))/2+"\n"+overlay+towerDetail;
        } else {
            tempOverlay = costOverlay + overlay + towerDetail;
        }
        
        if (Player.getInstance().getCash() - Player.getInstance().getTowerCost(type) >= 0) {
            unicodeFont.drawString(42 + (21 * 32), 416,tempOverlay);
        } else {
            if(!isPlaced) {
                gr.drawString("Not enough cash", position.x+32, position.y+32);
            }
            unicodeFont.drawString(42 + (21 * 32), 416,costOverlay, Color.red);
            unicodeFont.drawString(42 + (21 * 32), 431,overlay+towerDetail);
        }
    }

    public void findClosestCritter() {

        // If critters on the map
        if(critterList!=null) {

            // If no critters is locked on
            if(getTargetCritter()==null) {
                Critter tempTarget = new Critter("test");
                // Set initial position very far away from the map
                tempTarget.setPosition(new Vector2f(-1000f,-1000f));
                for(Critter enemy : critterList) {
                    float critterDistance = this.getPosition().distance(enemy.getPosition());
                    float tempDistance = this.getPosition().distance(tempTarget.getPosition());

                    // First check the critter is in range of the tower
                    // Then check if it is closer than the tempCritter
                    // Do this for all critters to eventually find the closest one
                    if(critterDistance < range) {
                        if(critterDistance < tempDistance) {
                            setTargetCritter(enemy);
                            tempTarget=enemy;
                        }
                    }
                    // Remnants of the multicolour lazers
                    // colourCounter++;
                }
            } else if (getTargetCritter()!=null) {
                // If the critter is out of range or dead, find a new target
                // Else shoot the
                if(this.getPosition().distance(getTargetCritter().getPosition()) >= range) {
                    setTargetCritter(null);
                } else if(getTargetCritter().isDead()||getTargetCritter().isDelete()) {
                    setTargetCritter(null);
                }
            }
        }
    }

    private void guiOverlay(Graphics gr) {
        
        costOverlay = "Cost : $" + Player.getInstance().getTowerCost(type)+"\n";
        overlay = "Range = " + (int)range +"\nDPS = " + (int)damagePerSec+"\n";
        
        if (!isPlaced) {
            if (circle != null) {
                gr.draw(circle);
            }
            renderTextOverlay(gr);
        } else if (isPlaced && mouseXTile == getTilePosition().x && mouseYTile == getTilePosition().y) {
            if (active) {
                if(!GameplayState.towerSelected) {
                    if (circle != null) {
                        gr.draw(circle);
                    }
                    renderTextOverlay(gr);
                }
            } else {
                renderTextOverlay(gr);
            }
            
            
        }
    }
    
    @Override
    public void setType(int type) {
        this.type=type;
        range = Settings.getInstance().getRange()[type];
        damagePerSec = Settings.getInstance().getBaseDPS()[type];
        circle.setRadius(range);
        circle.setCenterX(position.x+(GameplayState.TILESIZE/2));
        circle.setCenterY(position.y+(GameplayState.TILESIZE/2));
    }
    
    @Override
    public void setPosition(Vector2f position)
    {
        this.position = position;
        if(circle==null) {
            circle = new Circle(position.x+(GameplayState.TILESIZE/2),position.y+(GameplayState.TILESIZE/2),range);
        } else {
            circle.setCenterX(position.x+(GameplayState.TILESIZE/2));
            circle.setCenterY(position.y+(GameplayState.TILESIZE/2));
        }
    }

    @Override
    public void update(GameContainer gc, StateBasedGame sb, int delta)
    {
        Input i = gc.getInput();
        
        findClosestCritter();
        
        mouseXTile = (int) Math.floor((i.getAbsoluteMouseX() / GameplayState.TILESIZE));
        mouseYTile = (int) Math.floor((i.getAbsoluteMouseY() / GameplayState.TILESIZE));
        
        
        if(active && isPlaced && mouseXTile == getTilePosition().x && mouseYTile == getTilePosition().y){
            if(i.isKeyPressed(Input.KEY_DELETE) || i.isKeyPressed(Input.KEY_NUMPAD0)) {
                Player.getInstance().sellTower(type);
                this.killEntity();
            }
        }
        
        try {
            unicodeFont.loadGlyphs(100);
        } catch (SlickException ex) {
            Logger.getLogger(Tower.class.getName()).log(Level.SEVERE, null, ex);
        }

        for(Component component : components)
        {
            component.update(gc,sb,delta);
        }

    }

    @Override
    public void render(GameContainer gc, StateBasedGame sb, Graphics gr)
    {
    	
    	if(renderComponents != null) {
            for(RenderComponent sRenderComponent : renderComponents) {
            	sRenderComponent.render(gc, sb, gr);
            }
        }
        
        guiOverlay(gr);
        
    }

    /**
     * @param isPlaced the isPlaced to set
     */
    public void setIsPlaced(boolean isPlaced) {
        this.isPlaced = isPlaced;
    }

    /**
     * @return the type
     */
    @Override
    public int getType() {
        return type;
    }

    /**
     * @return the targetCritter
     */
    public Critter getTargetCritter() {
        return targetCritter;
    }

    /**
     * @param targetCritter the targetCritter to set
     */
    public void setTargetCritter(Critter targetCritter) {
        this.targetCritter = targetCritter;
    }

	public float getDamagePerSec() {
		return damagePerSec;
	}

	public boolean isActive() {
		return active;
	}

	public void setActive(boolean active) {
		this.active = active;
	}
    
}
